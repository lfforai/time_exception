#include <stdio.h>
#include <cfloat>
#include "kernel_example.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/util/work_sharder.h"
#include <string.h>
#include <vector>

using namespace tensorflow;
using namespace std;

REGISTER_OP("Example")
    .Attr("T:realnumbertype")
    .Input("input: T")
    .Input("k:int32")
    .Output("output:T")
    .Doc(R"doc(
Adds 1 to all elements of the tensor.
output: A Tensor.
  output = input + 1
)doc");

template <class T>
class LOF_one{
private:
	struct forsort{
	T value=0;
	int old_index=0;
	};
public:
	 int k;//k of LOF
	 int k_neighborhood_num;

	 int feature;//[num,feature]
     int num;//[num,feature]
	 const T* value;//all values
	 T* all_destances;

	 int p;

	 //distance between every value in LOF eachother
	 void all_distance(){
		  int totallen=static_cast<int>(num*num);
		  for(int i = 0; i < num; ++i) {
			  for(int j = i; j < num; ++j){
				  if(i==j){
					  all_destances[num*i+j]=(T)-1;
				  }else{
					  all_destances[num*i+j]=comput_distance(i,j);
					  all_destances[num*j+i]=all_destances[num*i+j];
				  }
			  }
		}
	  }

	  LOF_one(int p_index,int k_o,int* dims_o,T* value_o){
		  this->p=p_index;
		  this->k=k_o;
          this->value=value_o;
		  this->num=dims_o[0];
		  this->feature=dims_o[1];
		  this->all_destances=(T*)malloc((num*num*sizeof(T)));
		  this->all_distance();
	  };

	  //Euclidean distance
	  T comput_distance(int p,int o){
			  T result;
			  if(feature==1)
			  {result=(T)abs(value[p*feature]-value[o*feature]);}
			  else{
			  for(int i=0;i<feature;i++)
				   {result=result+(T)pow(value[p*feature+i]-value[p*feature+i],2);} //Euclidean distance
				   result=sqrt(result);
    		  }
			  return result;
	 };

	 inline T find_distance(int p,int o){
				  T result;
				  result=this->all_destances[p*num+o];
				  if(result==0)
					 result=(T)0.001;
			      return result;
	  };

	 void printvector(vector<int>* v){
		  for (auto iter = v->begin(); iter != v->end(); iter++)
		    {
		        cout << value[(*iter)] <<"|"<<*iter<< endl;
		    }
		  cout<<"--------------------"<<endl;
	 }

     // k-distance neighborhood of p：第k距离邻域
	 vector<int>* k_distance_neighborhood(int p_now)
	 {
		 vector<forsort> vectors;
		 for(int i = 0; i < num; ++i){
			 if(p_now!=i)
			 {forsort one;
			 one.value=this->all_destances[p_now*num+i];
			 one.old_index=i;
             vectors.push_back(one);
			 }
		 }

	     sort(vectors.begin(), vectors.end(),[](forsort &a,forsort &b){return a.value<b.value;});
	     vector<int>* result=new vector<int>;
	     int k_index=0;
	     while(true){
	    	 int k_inside=result->size();
	    	 if (k_inside<this->k){
	    		 result->push_back(vectors[k_index].old_index);
	    		 k_index+=1;
	    	 }else if(k_inside==this->k){
	    		 while(vectors[k_index-1].value==vectors[k_index].value && k_index<num-1){
	    			     result->push_back(vectors[k_index].old_index);
	    			     k_index+=1;
	    		   }
	    		  break;
	    	 }
	     }
		 return result;
	 }

	 //max of from's k neighborhood and distance between from and to
	 T reach_distance_k(int from,int to){
		vector<int>* from_neighborhood=k_distance_neighborhood(from);
		T k_distance=find_distance((*from_neighborhood)[(*from_neighborhood).size()-1],from);
		T dirct_distance=find_distance(from,to);
		T result=max(k_distance,dirct_distance);
		//free(from_neighborhood);
		return result;
	 }

	 //lrdk(P_in)
	 T local_reachability_density(int p_in){
			vector<int>* from_neighborhood=k_distance_neighborhood(p_in);
			int N_k_p=from_neighborhood->size();
			T result=0;
			for(int i=0;i<N_k_p;i++){
				result+=reach_distance_k((*from_neighborhood)[i],p_in);
			}
			//free(from_neighborhood);
			return ((T)(N_k_p))/(result);
	 }

	 //this->p局部离群因子
	 T local_outlier_factor(){
		 vector<int>* from_neighborhood=k_distance_neighborhood(this->p);
		 int N_k_p=from_neighborhood->size();
		 T lrdk_p=local_reachability_density(this->p);
		 T result=0;
		 for(int i=0;i<N_k_p;i++){
			 result+=local_reachability_density((*from_neighborhood)[i]);
		 }
		 result=(T)result/lrdk_p/((T)N_k_p);
		 //free(from_neighborhood);
         //printf("local_out=%f,p=%d \n",(float)result,this->p);
		 return result;
	 }
};

template class LOF_one<int>;
template class LOF_one<float>;
template class LOF_one<double>;

// OpKernel definition.
// template parameter <T> is the datatype of the tensors.
template <class Device, class T>
class ExampleOp : public OpKernel
{
  private:
  public:
  explicit ExampleOp(OpKernelConstruction* context) : OpKernel(context) {}
  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& input_tensor = context->input(0);
    const Tensor& k_tensor = context->input(1);

    // Create an output tensor
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(),
                                                     &output_tensor));//only shape no datatype

    // Do the computation,tensor is not too long
    OP_REQUIRES(context, input_tensor.NumElements() <= tensorflow::kint32max,
                errors::InvalidArgument("Too many elements in tensor"));

    // data should have 3 dimensions.  [20,1] or [20,2].......
    OP_REQUIRES(context, input_tensor.dims() == 2,
                    errors::InvalidArgument("data must be 2-dimensional"));

    int window_size=input_tensor.dim_size(0);//times windows
    int feature=input_tensor.dim_size(1);//x,y..

    //prepare data from tensor
    auto output_data=output_tensor->flat<T>().data();
    auto input_data=input_tensor.flat<T>().data();
    int k=(k_tensor.flat<int>().data())[0];

    const DeviceBase::CpuWorkerThreads& worker_threads =
            *(context->device()->tensorflow_cpu_worker_threads());

    //printf("worker_num:%d \n",worker_threads.num_threads);
    const int64 shard_cost=static_cast<int64>(input_tensor.NumElements()/worker_threads.num_threads);

    //LOF_one(int p_index,int k_o,int* dims_o,T*value_o);
    auto shard = [&k,&output_data, &input_data,window_size,feature]
                    (int64 start, int64 limit) {
    	 int dims[2]={window_size,feature};
    	 LOF_one<T>* oo_point;
    	 //printf("value:%d,%d \n",window_size,feature);
    	 for (int64 b = start; b < limit; ++b){
    		 oo_point=new LOF_one<T>((int)b,k,dims,(T*)input_data);
    		 cout<<"b:"<<oo_point->value[b]<<endl;
    		 oo_point->printvector(oo_point->k_distance_neighborhood((int)b));
    		 T v=oo_point->local_outlier_factor();
    		 output_data[(int)b]=(T)v;
    		 delete oo_point;
    	 };

    };

    Shard(worker_threads.num_threads, worker_threads.workers,
    		input_tensor.NumElements(), shard_cost, shard);
  }
};

//特例化
template <class T>
class ExampleOp<GPUDevice,T>: public OpKernel {
 public:
  explicit ExampleOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& input_tensor = context->input(0);

    // Create an output tensor
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(),
                                                     &output_tensor));

    // Do the computation.
    OP_REQUIRES(context, input_tensor.NumElements() <= tensorflow::kint32max,
                errors::InvalidArgument("Too many elements in tensor"));

//    ExampleFunctor<Device,T>()(
//        context->eigen_device<Device>(),
//        static_cast<int>(input_tensor.NumElements()),
//        input_tensor.flat<T>().data(),
//        output_tensor->flat<T>().data());
  }
};

//Register the CPU kernels.
#define REGISTER_CPU(T)                                          \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("Example").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      ExampleOp<CPUDevice, T>);

REGISTER_CPU(double);
REGISTER_CPU(float);
REGISTER_CPU(int32);

// Register the GPU kernels.
#ifdef GOOGLE_CUDA_O
#define REGISTER_GPU(T)                                          \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("Example").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
      ExampleOp<GPUDevice, T>);
REGISTER_GPU(double);
REGISTER_GPU(float);
REGISTER_GPU(int32);
#endif  // GOOGLE_CUDA_O



