#include "../init.hpp"
extern TimeRcd timeRcd;

void scan(clContext *clCxt,cl_mem &ginput,cl_mem &goutput,Plan *plan,int elemnum)
{
    cl_mem gadsys;
    int steplength =  ((plan->registergroup + plan->localmemgroup) * plan->cta) / plan->vectorlength; 
    int taillen = elemnum - steplength * plan->vectorlength * plan->localthread * plan->workgroup, tailgroup;
    tailgroup = (taillen+2047)>>11;
    int adsyslen = plan->workgroup+tailgroup+2;
    int *adsys = new int[adsyslen];
     
    memset(adsys,0,adsyslen*sizeof(int));
    create(clCxt,&gadsys,adsyslen*sizeof(int)); 
    upload(clCxt,(void *)adsys,&gadsys,adsyslen * sizeof(int));
    int registersize = plan->registergroup * plan->cta;
    int localmemsize = plan->localmemgroup * plan->cta;
    char build_options[200];
    if(plan->coalesced==1){
  	    sprintf(build_options ,
        "-D NB_VEC_%d -D NB_L%d -D NB_G%d -D NB_CTA_%d -D STEP_NUM=%d -D NB_REG_GRP=%d -D NB_REG_SIZE=%d -D NB_LOCAL_GRP=%d -D NB_LOCAL_SIZE=%d",
        plan->vectorlength,plan->localthread,plan->workgroup,plan->cta,steplength,plan->registergroup,
        registersize,plan->localmemgroup,localmemsize + 1);
    }
    else{
  	    sprintf(build_options ,
        "-D NB_VEC_%d -D NB_L%d -D NB_G%d -D NB_CTA_%d -D STEP_NUM=%d -D NB_REG_GRP=%d -D NB_REG_SIZE=%d -D NB_LOCAL_GRP=%d -D NB_LOCAL_SIZE=%d",
        plan->vectorlength,plan->localthread,plan->workgroup,plan->cta,steplength,plan->registergroup,
        registersize,plan->localmemgroup,localmemsize);
    }
    cout<<build_options<<endl;
    vector<pair<size_t ,const void *> > args;
    args.push_back( make_pair( sizeof(cl_mem) , (void *)&ginput ));
    args.push_back( make_pair( sizeof(cl_mem) , (void *)&gadsys ));
    args.push_back( make_pair( sizeof(cl_mem) , (void *)&goutput ));
    args.push_back( make_pair( sizeof(cl_int) , (void *)&plan->workgroup ));
    size_t globalthreads[3] = {plan->localthread * plan->workgroup,1,1};
    size_t localthreads[3] = {plan->localthread,1,1};
    
    timeRcd.kerneltime = 0;
    timeRcd.totaltime = 0;
    executeKernel("scan.cl","scan",args,globalthreads,localthreads,build_options,clCxt);

    if(taillen!=0){
  	    sprintf(build_options ,
        "-D NB_VEC_TAIL -D NB_L64 -D NB_G%d -D NB_CTA_16 -D STEP_NUM=32 -D NB_REG_GRP=1 -D NB_REG_SIZE=16 -D NB_LOCAL_GRP=1 -D NB_LOCAL_SIZE=17",
        tailgroup);
        args.clear();
        args.push_back( make_pair( sizeof(cl_mem) , (void *)&ginput));
        args.push_back( make_pair( sizeof(cl_mem) , (void *)&gadsys));
        args.push_back( make_pair( sizeof(cl_mem) , (void *)&goutput));
        args.push_back( make_pair( sizeof(cl_int) , (void *)&elemnum));
        args.push_back( make_pair( sizeof(cl_int) , (void *)&plan->workgroup));
        args.push_back( make_pair( sizeof(cl_int) , (void *)&tailgroup));
        args.push_back( make_pair( sizeof(cl_int) , (void *)&taillen));
        globalthreads[0] = 64 * tailgroup;
        localthreads[0] = 64;
        //timeRcd.kerneltime = 0;
        //timeRcd.totaltime = 0;
        executeKernel("scan.cl","scantail",args,globalthreads,localthreads,build_options,clCxt);
    }

//#define PRINT_A    
#if defined PRINT_A   
    download(clCxt,&gadsys,(void *)adsys,adsyslen*sizeof(int));
    for(int i=0,k=1;i< adsyslen;k++)
    {
        cout <<setiosflags(ios::left) << "line:"<<k<<"   ";
        for(int j=0;j < 10 && i< adsyslen;j++,i++)
           cout << setiosflags(ios::left) << setw(10) << adsys[i];
        cout << endl;
    }
#endif    
    delete adsys; 
    clReleaseMemObject(gadsys);

}

///////////////////////////////////// scan CPU///////////////////////////////////////
void cpuScan(int *input,int elemnum,int *output)
{
    struct timeval vstart,vend;
    double start,end;
    gettimeofday(&vstart,NULL);
    start=(double)vstart.tv_sec*1000000 + (double)vstart.tv_usec;
    int sum=0;
    for(int i = 0;i<elemnum;i++)
    {
        sum += input[i];
        output[i] = sum;
    }
    gettimeofday(&vend,NULL);
    end=(double)vend.tv_sec*1000000 + (double)vend.tv_usec;
    timeRcd.min_cputime = (end-start)/1000;
//    printf("cpu scan time:%lf\r\n",(end - start)/1000);
}

int check(int* c,int *g,int elemnum)
{
    int flag=1;
    for(int i=0;i<elemnum;i++)
    {
        if(c[i]!=g[i])
        {
            cout<<"Error:"<<i<<"    CPU "<<c[i]<<" GPU "<<g[i]<<endl;
            flag=0;
            return 0;
        }
    }
    if(flag=1) cout<<"Result OK!"<<endl<<endl;
}

void test_scan(clContext *clCxt,int elemnum)
{
    int * input = (int*)malloc(elemnum * sizeof(int)),*output;
    output = (int *)malloc(elemnum*sizeof(int));
    int range =5;
    arrayinit(input,elemnum,range);
    int csum=0,gsum=0;
    Plan *plan=(Plan *)malloc(sizeof(Plan));
    memset(output,0,elemnum*sizeof(int));
    cl_mem ginput, goutput; 
    create(clCxt,&ginput,elemnum * sizeof(int));
    create(clCxt,&goutput,elemnum * sizeof(int));
    upload(clCxt,(void *)input,&ginput,elemnum * sizeof(int));
    int *cpuoutput;

//#define PRINT_R 
#define CHECK
#define PERFORMANCE
#if defined PERFORMANCE
    //coalesced performance 
    for(int vec=2;vec>=1;vec--)
        for(int lt=128;lt<=256;lt<<=1)
            for(int ct=8;ct<=32;ct<<=1)
                for(int regp=0;regp<=8&&regp*ct<240;regp++)
                    for(int logp=1;logp<=(8*1024)/(lt*ct);logp++)
                    {
                        plan->localthread = lt; 
                        plan->workgroup = elemnum/(lt*(regp+logp)*ct);
                        plan->cta = ct;
                        plan->registergroup = regp;
                        plan->localmemgroup = logp;
                        plan->vectorlength = vec;
                        plan->coalesced = 1;
                        scan(clCxt,ginput,goutput,plan,elemnum);
                        double avg_k_time=0.0,avg_t_time=0.0;
                        int run_times=10;
                        for(int i=0;i<run_times;i++){
                            scan(clCxt,ginput,goutput,plan,elemnum);
                            avg_k_time=avg_k_time + timeRcd.kerneltime;
                            avg_t_time=avg_t_time + timeRcd.totaltime;
                        }
                        download(clCxt,&goutput,(void *)output,elemnum * sizeof(int));
                        #if defined CHECK
                            cpuoutput = (int *)malloc(elemnum*sizeof(int));
                            cpuScan(input,elemnum,cpuoutput);
                            check(cpuoutput,output,elemnum);
                        #endif
                        avg_k_time=avg_k_time/run_times;
                        avg_t_time=avg_t_time/run_times;
                        if(avg_k_time < timeRcd.min_kerneltime && avg_k_time > 0) timeRcd.min_kerneltime = avg_k_time;
                        if(avg_t_time < timeRcd.min_totaltime && avg_t_time > 0) timeRcd.min_totaltime = avg_t_time;
                        cout<<"avg_k_time:"<<avg_k_time<<"  avg_t_time:"<<avg_t_time<<"  min time:"<<timeRcd.min_kerneltime<<endl<<endl;
                    }
#else    
    plan->localthread = 128; 
    plan->cta = 32;
    plan->registergroup = 1;
    plan->localmemgroup = 1;
    plan->vectorlength = 2;
    plan->coalesced = 1;
    plan->workgroup = elemnum / (plan->localthread * (plan->registergroup + plan->localmemgroup) * plan->cta);
    scan(clCxt,ginput,goutput,plan,elemnum);
    double avg_k_time=0.0,avg_t_time=0.0;
    for(int i=0;i<10;i++){
        scan(clCxt,ginput,goutput,plan,elemnum);
        avg_k_time=avg_k_time + timeRcd.kerneltime;
        avg_t_time=avg_t_time + timeRcd.totaltime;
    }
    avg_k_time=avg_k_time/10;
    avg_t_time=avg_t_time/10;
    if(avg_k_time < timeRcd.min_kerneltime && avg_k_time > 0) timeRcd.min_kerneltime = avg_k_time;
    if(avg_t_time < timeRcd.min_totaltime && avg_t_time > 0) timeRcd.min_totaltime = avg_t_time;
    cout<<"Elements number:"<<elemnum<<"    min_k_time:"<<avg_k_time<<"  min_t_time:"<<avg_t_time<<endl<<endl;
#endif
    download(clCxt,&goutput,(void *)output,elemnum * sizeof(int));

#if defined CHECK
    cpuoutput = (int *)malloc(elemnum*sizeof(int));
    cpuScan(input,elemnum,cpuoutput);
    check(cpuoutput,output,elemnum);
#endif

#if defined PRINT_R
    for(int i=0,k=1;i< elemnum;k++)
    {
        cout <<setiosflags(ios::left) << "line:"<<k<<"   ";
        for(int j=0;j < 10 && i< elemnum;j++,i++)
           cout << setiosflags(ios::left) << setw(10) << output[i];
        cout << endl;
    }
#endif    
    clReleaseMemObject(ginput);
    clReleaseMemObject(goutput);
    free(input);
    free(output);
    free(cpuoutput);
    return;
}

int main()
{
    int elemnum = 1024 * 1024 * 16;
    clContext clCxt;
    getClContext(&clCxt);
    timeRcd.min_kerneltime=1000000;
    timeRcd.min_totaltime=1000000;
    test_scan(&clCxt,elemnum);
    printf("kernel min kernel time:%lf     ",timeRcd.min_kerneltime);
    printf("kernel min total time:%lf     \n",timeRcd.min_totaltime);
    return 0;
}
