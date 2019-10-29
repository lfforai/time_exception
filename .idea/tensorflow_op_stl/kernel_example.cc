#include <stdio.h>
#include <cfloat>
#include "kernel_example.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/util/work_sharder.h"
#include </usr/include/eigen3/Eigen/Dense>
#include <string.h>
#include <vector>

using namespace tensorflow;
using namespace std;
//using namespace Eigen;

REGISTER_OP("Example")
    .Attr("T:realnumbertype")
    .Input("input: T")
    .Input("n_s:int32")//第二步骤样本loess平滑参数n(s)
    .Input("n_p:int32")//第三步骤移动平滑长度np和第二步样本序列个数
    .Input("n_l:int32")//第三步骤移动平滑长度np
    .Input("n_t:int32")//第三，六步骤loess平滑参数nt
    .Input("r: T")//loess r,exp(-（x-x1）/2×r^2)
    .Input("wai: int32")//外循环次数
    .Input("nei: int32")//内循环次数
    .Output("outputt:T")
    .Output("outputs:T")
    .Output("outputr:T");

template <class T>
class loess{
private:
public:
	 T* x;
	 int size;
	 T r;//exp(-1*(x1-x)/(2*r^2))
	 T* w;//weight
	 const T* y;//原值
	 T y_pr;//预测值 bais+wm8*x_mid=y_pr;
	 int index;

	 T param_loess[2];

     void weight_init(T* neigh_w){
          for(int i = 0; i < size; ++i){
		     w[i]=exp(-1*abs(x[i]-index)/(2.0*r*r))*neigh_w[i];
		  }
     }

     loess(const T* y_o,int index_o,int size_o,T r_o,T* neigh_w){
    	 this->x=(T*)malloc(size_o*sizeof(T));
    	 for(int i=0;i<size_o;++i){
    		 this->x[i]=(T)i;
    	 }
    	 this->y=y_o;
    	 this->size=size_o;
    	 this->r=r_o;
    	 this->w=(T*)malloc(size*sizeof(T));
    	 this->index=index_o;
    	 this->weight_init(neigh_w);
     }

     void loess_nh(){
          T* X=(T*)malloc(2*size*sizeof(T));
          for(int i = 0; i < size;++i)
             {  X[i*2]=(T)1;
            	X[i*2+1]=(T)x[i];
		     }

		  if(typeid(T)==typeid(float)){
                //w=(x^t*w*x)^I*x^t*w*y
			    Eigen::MatrixXf X_mat=Eigen::Map<Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>>(X,this->size,2);
			    Eigen::MatrixXf Y_mat=Eigen::Map<Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>>((T*)this->y,this->size,1);
				vector<float> stdvec(w,w+size);
				Eigen::Map<Eigen::VectorXf> eigVec1(stdvec.data(), stdvec.size());
				//cout<<eigVec1<<endl;
				//cout<<"-----"<<endl;
				Eigen::MatrixXf W_mat=eigVec1.asDiagonal();
				Eigen::MatrixXf param =(X_mat.transpose()*W_mat*X_mat).inverse()*X_mat.transpose()*W_mat*Y_mat;

				float* param_p=param.data();
				float bais=param_p[0];
				this->param_loess[0]=bais;
				float wm=param_p[1];
				this->param_loess[1]=(T)wm;
				//cout<<"bais:"<<bais<<"|"<<wm<<endl;
				this->y_pr=bais+wm*((T)(this->index));

		  }else{
			cout<<"type"<<typeid(T).name()<<"not support!"<<endl;
			exit(0);
		  }

     }

     //返回index=-1，向前推进
     T pre_frist(){
			return this->param_loess[0]+(this->param_loess[1])*((T)(-1));
		}

     //返回index=size，向后推进
     T pre_last(){
    		return this->param_loess[0]+(this->param_loess[1])*((T)(size));
		}
};

template class loess<float>;

// OpKernel definition.
// template parameter <T> is the datatype of the tensors.
template <class Device, class T>
class ExampleOp : public OpKernel
{
  private:
	void tensor_sub(T*a,T*b,T*c,int len){
		for (int i = 0; i < len; ++i) {
			c[i]=a[i]-b[i];
		}
	}

	T inline B_fun(T u){
		T result=(T)(0);
		if(u>=(T)0 && u<(T)(1)){
			result=pow(1-u*u,2);
		}
		return result;
	}

  public:
  explicit ExampleOp(OpKernelConstruction* context) : OpKernel(context) {}
  void Compute(OpKernelContext* context) override
  {
    //// Grab the input tensor
	const Tensor& input_tensor = context->input(0);
    int totallenght=(int)input_tensor.NumElements();
   // Do the computation,tensor is not too long
	OP_REQUIRES(context, input_tensor.NumElements() <= tensorflow::kint32max,
				errors::InvalidArgument("Too many elements in tensor"));

	// data should have 2 dimensions.  [20,1] or [20,2].......
	OP_REQUIRES(context, input_tensor.dims() == 2,
					errors::InvalidArgument("data must be 2-dimensional"));

	//ns np nl nt r
	const Tensor& ns_tensor = context->input(1);//第二步骤loess平滑参数n(s)
	const Tensor& np_tensor = context->input(2);//第三步骤移动平滑长度np
	const Tensor& nl_tensor = context->input(3);//第三步骤loess平滑参数n1
	const Tensor& nt_tensor = context->input(4);//第三步骤loess平滑参数n1
	const Tensor& r_tensor = context->input(5);//第三步骤loess平滑参数n1
	const Tensor& wai_tensor = context->input(6);//第三步骤loess平滑参数n1
	const Tensor& nei_tensor = context->input(7);//第三步骤loess平滑参数n1

    int ns=(int)(ns_tensor.flat<int>().data()[0]);//必须是奇数
    int np=(int)(np_tensor.flat<int>().data()[0]);
    int nl=(int)(nl_tensor.flat<int>().data()[0]);//必须是奇数
    int nt=(int)(nt_tensor.flat<int>().data()[0]);//必须是奇数
    T r=r_tensor.flat<T>().data()[0];
    int wai=wai_tensor.flat<int>().data()[0];
    int nei=nei_tensor.flat<int>().data()[0];

    // Do the computation,tensor must be odd
    OP_REQUIRES(context, ns%2==1,
                errors::InvalidArgument("ns must be odd！"));
    OP_REQUIRES(context, nl%2==1,
                  errors::InvalidArgument("nl must be odd！"));
    OP_REQUIRES(context, nt%2==1,
                   errors::InvalidArgument("nt must be odd!"));
    OP_REQUIRES(context, ((int)(totallenght/np))*np==totallenght,
                       errors::InvalidArgument("被检测数组的长度是必须是周期np的整数倍！"));

    //output tensor setup
    int dims[1];
    dims[0]=(int)input_tensor.NumElements();
    TensorShape output_shape_STR;
    TensorShapeUtils::MakeShape(dims, 1, &output_shape_STR);

    //Create an output tensor 趋势T
    Tensor* outputT_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape_STR,
                                                     &outputT_tensor));//only shape no datatype
    // Create an output tensor 季节周期S
    Tensor* outputS_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, output_shape_STR,
                                                       &outputS_tensor));//only shape no datatype

    // Create an output tensor 随机R
    Tensor* outputR_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(2, output_shape_STR,
                                                       &outputR_tensor));//only shape no datatype

    //prepare data from tensor
    auto outputs=outputS_tensor->flat<T>().data();
    auto outputt=outputT_tensor->flat<T>().data();
    auto outputr=outputR_tensor->flat<T>().data();

    //set kernel
    const DeviceBase::CpuWorkerThreads& worker_threads =
            *(context->device()->tensorflow_cpu_worker_threads());
    const int64 shard_cost=static_cast<int64>(totallenght/worker_threads.num_threads);

	//neighborhood_weight 外循环
	T* neig_w=(T*)malloc((totallenght)*sizeof(T));//for step6
	T* neig_w_sub;
	T* neig_w_3=(T*)malloc((totallenght)*sizeof(T));//for step3 loess
	for(int n = 0; n < totallenght; ++n) {
		neig_w[n]=(T)(1);
		neig_w_3[n]=(T)(1);
	}

	//第一步
    auto Y_v=input_tensor.flat<T>().data();
    T* T_v=(T*)malloc(totallenght*sizeof(T));

    //第二步
    T* C_v=(T*)malloc(totallenght*sizeof(T));
    T* C_v_sub;//子列
    T* C_v_sub_loess;//子列for loess
    T* C_v_2np=(T*)malloc((totallenght+2*np)*sizeof(T));

    //EMA移动平均
    T* C_v_2np_EMA1=(T*)malloc((totallenght+np+1)*sizeof(T));
    memset(C_v_2np_EMA1,0,(totallenght+np+1)*sizeof(T));
    T* C_v_2np_EMA2=(T*)malloc((totallenght+2)*sizeof(T));
    memset(C_v_2np_EMA2,0,(totallenght+2)*sizeof(T));
    T* C_v_2np_EMA3=(T*)malloc((totallenght)*sizeof(T));
    memset(C_v_2np_EMA3,0,(totallenght)*sizeof(T));

    int C_v_sub_num=0;//子列元素个数
    T* C_v_o=(T*)malloc(totallenght*sizeof(T));//保留在第3步使用

    //第三步
    T* L_v=(T*)malloc((totallenght)*sizeof(T));
    //第四步
    T* S_v=(T*)malloc((totallenght)*sizeof(T));
    //第五步
    T* Y_v_sub_S_v=(T*)malloc((totallenght)*sizeof(T));

    //loess第2步,
    T before;
    T last;
    auto shard_loess2 = [&C_v_sub,&C_v_sub_loess,&ns,&r,&C_v_sub_num,&before,&last,&neig_w_sub]
                    (int64 start, int64 limit) {
    	 int block_size=(int)(ns/2);//ns=5，
    	 if (ns>C_v_sub_num){
    		 cout<<"ns大于类loess数组的长度！"<<endl;
    		 exit(0);
    	 }

    	 for (int64 b = start; b < limit; ++b){
    		 if(b-block_size>=0 && b+block_size<C_v_sub_num)
    		 {loess<T>* oo_point=new loess<T>(C_v_sub+b-block_size,block_size,ns,r,neig_w_sub+b-block_size);
    		  oo_point->loess_nh();
    		  C_v_sub_loess[(int)b]=oo_point->y_pr;
    		  if(b==0){
    			  before=oo_point->pre_frist();
    		  }
    		  if(b==C_v_sub_num-1){
    			  last=oo_point->pre_last();
    		  }

             }else if(b-block_size<0){
    		  loess<T>* oo_point=new loess<T>(C_v_sub,b,ns,r,neig_w_sub);
    		  oo_point->loess_nh();
    	      C_v_sub_loess[(int)b]=oo_point->y_pr;
    		  if(b==0){
    			  before=oo_point->pre_frist();
    		  }
    		  if(b==C_v_sub_num-1){
    			  last=oo_point->pre_last();
    		  }

    		 }else if(b+block_size>=C_v_sub_num){
    	      loess<T>* oo_point=new loess<T>(C_v_sub+C_v_sub_num-ns,block_size-(C_v_sub_num-b),ns,r,neig_w_sub+C_v_sub_num-ns);
    	      oo_point->loess_nh();
    	      C_v_sub_loess[(int)b]=oo_point->y_pr;
    		  if(b==0){
    			  before=oo_point->pre_frist();
    		  }
    		  if(b==C_v_sub_num-1){
    			  last=oo_point->pre_last();
    		  }

    		 }
    	 };
    };

    //loess第3步
    auto shard_loess3 = [&C_v_2np_EMA3,&L_v,&nl,&r,&totallenght,&neig_w_3]
                    (int64 start, int64 limit) {
		 if (nl>totallenght){
			 cout<<"n1大于类loess数组的长度！"<<endl;
			 exit(0);
		 }
    	 int block_size=(int)(nl/2);//ns=5，
    	 for (int64 b = start; b < limit; ++b){
    		 if(b-block_size>=0 && b+block_size<totallenght)
    		 {loess<T>* oo_point=new loess<T>(C_v_2np_EMA3+b-block_size,block_size,nl,r,neig_w_3+b-block_size);
    		  oo_point->loess_nh();
    		  L_v[(int)b]=oo_point->y_pr;
             }else if(b-block_size<0){
    		  loess<T>* oo_point=new loess<T>(C_v_2np_EMA3,b,nl,r,neig_w_3);
    		  oo_point->loess_nh();
    		  L_v[(int)b]=oo_point->y_pr;
    		 }else if(b+block_size>=totallenght){
    	      loess<T>* oo_point=new loess<T>(C_v_2np_EMA3+totallenght-nl,block_size-(totallenght-b),nl,r,neig_w_3+totallenght-nl);
    	      oo_point->loess_nh();
    	      L_v[(int)b]=oo_point->y_pr;
    		 }
    	 };
    };

    //loess第6步
    auto shard_loess6 = [&Y_v_sub_S_v,&T_v,&nt,&r,&totallenght,&neig_w]
                    (int64 start, int64 limit) {
	 if (nt>totallenght){
			 cout<<"nt大于类loess数组的长度！"<<endl;
			 exit(0);
		 }
   	 int block_size=(int)(nt/2);//ns=5，
   	 for (int64 b = start; b < limit; ++b){
   		 if(b-block_size>=0 && b+block_size<totallenght)
   		 {loess<T>* oo_point=new loess<T>(Y_v_sub_S_v+b-block_size,block_size,nt,r,neig_w+b-block_size);
   		  oo_point->loess_nh();
   		  T_v[(int)b]=oo_point->y_pr;
            }else if(b-block_size<0){
   		  loess<T>* oo_point=new loess<T>(Y_v_sub_S_v,b,nt,r,neig_w);
   		  oo_point->loess_nh();
   		  T_v[(int)b]=oo_point->y_pr;
   		 }else if(b+block_size>=totallenght){
   	      loess<T>* oo_point=new loess<T>(Y_v_sub_S_v+totallenght-nt,block_size-(totallenght-b),nt,r,neig_w+totallenght-nt);
   	      oo_point->loess_nh();
   	      T_v[(int)b]=oo_point->y_pr;
   		 }
   	  }
    };

    //第3步骤移动平均,执行2次
    auto shard_EMA_np1 = [&np,&C_v_2np,&C_v_2np_EMA1]
                     (int64 start, int64 limit) {
    	 //printf("value:%d,%d \n",window_size,feature);
    	 for (int64 b = start; b < limit; ++b){
    		 //cout<<"b:"<<start<<"|"<<(limit-start)<<endl;
    		 for(int j=0;j<np;++j)
    		    {
    			 C_v_2np_EMA1[(int)b]+=C_v_2np[(int)(b+j)];
			    }
    		 C_v_2np_EMA1[(int)b]=C_v_2np_EMA1[(int)b]/((T)np);
    	 };
    };

    auto shard_EMA_np2 = [&np,&C_v_2np_EMA1,&C_v_2np_EMA2]
                     (int64 start, int64 limit) {
    	 //printf("value:%d,%d \n",window_size,feature);
    	 for (int64 b = start; b < limit; ++b){
    		 //cout<<"b:"<<start<<"|"<<(limit-start)<<endl;
    		 for(int j=0;j<np;++j)
    		    {
    			 C_v_2np_EMA2[(int)b]+=C_v_2np_EMA1[(int)(b+j)];
			    }
    		 C_v_2np_EMA2[(int)b]=C_v_2np_EMA2[(int)b]/((T)np);
    	 };
    };

    auto shard_EMA_np3 = [&C_v_2np_EMA2,&C_v_2np_EMA3]
                     (int64 start, int64 limit) {
    	 //printf("value:%d,%d \n",window_size,feature);
    	 for (int64 b = start; b < limit; ++b){
    		 //cout<<"b:"<<start<<"|"<<(limit-start)<<endl;
    		 for(int j=0;j<3;++j)
    		    {
    			 C_v_2np_EMA3[(int)b]+=C_v_2np_EMA2[(int)(b+j)];
			    }
    		 C_v_2np_EMA3[(int)b]=C_v_2np_EMA3[(int)b]/((T)3);
    	 };
    };

    C_v_sub_num=(int)(totallenght/np);//step2 每个子列元素个数
	cout<<"每个子列元素个数为："<<C_v_sub_num<<endl;
    C_v_sub=(T*)malloc(C_v_sub_num*sizeof(T));
    C_v_sub_loess=(T*)malloc(C_v_sub_num*sizeof(T));
    neig_w_sub=(T*)malloc(C_v_sub_num*sizeof(T));

    for (int w = 0; w<wai; ++w){//外循环
		//内循环
		for (int i = 0; i<nei; ++i){
			//step 1
			if(i==0)
			{memcpy(C_v,Y_v,totallenght*sizeof(T));}
			else{tensor_sub((T*)(Y_v),T_v,C_v,totallenght);}

			//setp 2
			for(int j = 0; j < np; ++j){
				//C_v=>C_v_sub
				for(int k = 0; k < C_v_sub_num; ++k){
					 C_v_sub[k]=C_v[j+np*k];
					 neig_w_sub[k]=neig_w[j+np*k];
				}

				//C_v_sub=>C_v_sub_loess
				Shard(worker_threads.num_threads,worker_threads.workers,
						C_v_sub_num,shard_cost,shard_loess2);

				//C_v_sub_loess=>C_v_2np
				C_v_2np[j]=before;
				C_v_2np[totallenght+np+j]=last;
				for(int k = 0; k < C_v_sub_num; ++k){
					C_v_2np[np*(k+1)+j]=C_v_sub_loess[k];
				}
			}
//			if(i==0)
//			  {for(int w=600;w<650;++w)
//			    cout<<"|"<< C_v_2np[w]<<endl;}
			//step3
			memcpy(C_v_o,C_v_2np+np,totallenght*sizeof(T));
			Shard(worker_threads.num_threads, worker_threads.workers,
						totallenght+(np+1), shard_cost, shard_EMA_np1);

			Shard(worker_threads.num_threads, worker_threads.workers,
						totallenght+2, shard_cost, shard_EMA_np2);

			Shard(worker_threads.num_threads, worker_threads.workers,
						totallenght, shard_cost, shard_EMA_np3);

			Shard(worker_threads.num_threads, worker_threads.workers,
						totallenght, shard_cost, shard_loess3);

			//step4
			tensor_sub(C_v_o,L_v,S_v,totallenght);
			//step5
			tensor_sub((T*)(Y_v),S_v,Y_v_sub_S_v,totallenght);
			//step6
			Shard(worker_threads.num_threads, worker_threads.workers,
												totallenght, shard_cost, shard_loess6);

			tensor_sub((T*)(Y_v),S_v,outputr,totallenght);
			tensor_sub(outputr,T_v,outputr,totallenght);
		}
	    //R
		vector<T> median_v;
		for(int n = 0; n < totallenght; ++n) {
			median_v.push_back(abs(outputr[n]));
		}
		sort(median_v.begin(),median_v.end());
		T med_R;
		if(median_v.size()%2==1){//odd
		   int m_index=(int)(median_v.size()/2);
		   med_R=median_v[m_index];
		}else{
			 int m_index=(int)(median_v.size()/2);
			 med_R=(T)((median_v[m_index-1]+median_v[m_index])/((T)2));
		}

		T h=((T)6)*med_R;
		for(int n = 0; n < totallenght; ++n) {
		   neig_w[n]=B_fun(abs(outputr[n])/h);
		}
    }//外循环
	memcpy(outputs,S_v,(totallenght)*sizeof(T));
	memcpy(outputt,T_v,(totallenght)*sizeof(T));
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

//REGISTER_CPU(double);
REGISTER_CPU(float);
//REGISTER_CPU(int32);

//Register the GPU kernels.
#ifdef GOOGLE_CUDA_O
#define REGISTER_GPU(T)                                          \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("Example").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
      ExampleOp<GPUDevice, T>);
//REGISTER_GPU(double);
REGISTER_GPU(float);
//REGISTER_GPU(int32);
#endif  // GOOGLE_CUDA_O



