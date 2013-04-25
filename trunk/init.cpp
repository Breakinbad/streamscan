#include "init.hpp"
TimeRcd timeRcd;
void getClContext(clContext *clCxt)
{
    cl_int ret;
    cl_uint num_platforms;
    ret = clGetPlatformIDs(0, NULL, &num_platforms);
    if(ret != CL_SUCCESS){
        cout << "Failed to get platform number. error code:"<< ret << endl;
        return ;
    }
    clCxt->num_platforms = num_platforms;
	  cl_platform_id *platforms = (cl_platform_id *)malloc(clCxt->num_platforms*sizeof(cl_platform_id));
    ret = clGetPlatformIDs(clCxt->num_platforms,platforms,NULL);
    if(ret != CL_SUCCESS){
        cout << "Failed to get platform ID. error code:"<< ret << endl;
        return ;
    }
    cout << "platforms_num:"<<num_platforms<<endl;
    clCxt->platform_id = platforms[1];
    cl_context_properties cps[3] = { CL_CONTEXT_PLATFORM,(cl_context_properties) (clCxt->platform_id), 0};
    cl_context_properties *cprops = (NULL == clCxt->platform_id) ? NULL :cps;
	  //create OpenCL context
    clCxt->context = clCreateContextFromType(cprops, CL_DEVICE_TYPE_GPU, NULL, NULL, &ret);
    if(ret != CL_SUCCESS)
    {
        cout << "Failed to create context. error code:"<< ret << endl;
		    return;
    }
    //get device id from context
    size_t device_num;
    ret = clGetContextInfo(clCxt->context, CL_CONTEXT_DEVICES, 0, NULL, &device_num);
    if(ret != CL_SUCCESS)
    {
        cout << "Failed to get device number. error code:"<< ret << endl;
        return ;
    }
    cl_device_id *devices=(cl_device_id *) malloc(device_num);
    ret = clGetContextInfo(clCxt->context, CL_CONTEXT_DEVICES, device_num, devices, NULL);
    if(ret != CL_SUCCESS)
    {
        cout << "Failed to get device ID. error code:"<< ret << endl;
        return ;
    }
    clCxt->device_id = devices[0];
    clCxt->num_devices = device_num;
    clCxt->command_queue = clCreateCommandQueue(clCxt->context, clCxt->device_id, CL_QUEUE_PROFILING_ENABLE, &ret);
    if(ret != CL_SUCCESS){
        cout << "Failed to create command queue. error code:"<< ret << endl;
        return ;
    }
}

void releaseContext(clContext *clCxt)
{
    cl_int ret;
    ret = clReleaseCommandQueue(clCxt->command_queue);
    if(ret != CL_SUCCESS){
        cout << "Failed to release command queue. error code:"<< ret << endl;
        return ;
    }
    ret = clReleaseContext(clCxt->context);
    if(ret != CL_SUCCESS){
        cout << "Failed to release context. error code:"<< ret << endl;
        return ;
    }
    for(int i = 0;i < clCxt->program.size();i ++)
    {
        ret = clReleaseProgram(*(clCxt->program[i].second));
        if(ret != CL_SUCCESS){
            cout << "Failed to release program. error code:"<< ret << endl;
            return ;
        }
    }
}

void executeKernel(string source,string kernelName, vector< pair<size_t,const void *> > args,
                         const size_t * globalthreads,const size_t* localthreads,
                         char * build_options,clContext *clCxt)
{
    cl_program *program = NULL;
    cl_kernel kernel = NULL;
    cl_int ret;
    FILE *fp;
    char *source_str;
    size_t source_size;
    fp = fopen(source.c_str(), "r");
    if (!fp) {
        printf("Failed to load kernel file.\n");
        exit(1);
    }
    source_str = (char*)malloc(MAX_SOURCE_SIZE);
    source_size = fread( source_str, 1, MAX_SOURCE_SIZE, fp);
    fclose( fp );
    string build_options_str(build_options);
    for(int i=0;i<clCxt->program.size();i++)
    {
        if(build_options_str==clCxt->program[i].first)
            program = clCxt->program[i].second;
    } 
    if(program == NULL)
    {
        printf("Compile the source code.\n");
        program = (cl_program *)malloc(sizeof(cl_program));
        *program = clCreateProgramWithSource(clCxt->context, 1, (const char **)&source_str,
	  		                                    (const size_t *)&source_size, &ret);
        if(ret != CL_SUCCESS){
            cout << "Failed to create Program. error code:"<< ret << endl;
            return ;
        }
        ret = clBuildProgram(*program,1,&(clCxt->device_id), build_options, NULL,NULL);
        if(ret != CL_SUCCESS){
            cout << "Failed to build Program. error code:"<< ret << endl;
        }
        char *buildLog = NULL;
        size_t buildLogSize = 0;
        clGetProgramBuildInfo(*program,clCxt->device_id,CL_PROGRAM_BUILD_LOG,buildLogSize,
                                          buildLog,&buildLogSize);
        buildLog = new char[buildLogSize];
        memset(buildLog,0,buildLogSize);
        clGetProgramBuildInfo(*program,clCxt->device_id,
                              CL_PROGRAM_BUILD_LOG,buildLogSize,buildLog,NULL);
        if(ret != CL_SUCCESS){
            cout << "\n\t\t\tBUILD LOG\n";
            cout << buildLog << endl;
            return;
        }
        delete buildLog;
        clCxt->program.push_back( make_pair( build_options_str , program ));
    }
    
    kernel = clCreateKernel(*program, kernelName.c_str(), &ret);
    if(ret != CL_SUCCESS){
        cout << "Failed to create Kernel. error code:"<< ret << endl;
        return ;
    }
    for(int i = 0;i < args.size();i ++)
    {
        ret = clSetKernelArg(kernel,i,args[i].first,args[i].second);
        if(ret != CL_SUCCESS){
            cout << "Failed to set Arg.Arg:"<< i <<", error code:"<< ret  << endl;
            return ;
        }
    }
    cl_event time_event;
    cl_ulong queued,start,end;
    ret = clEnqueueNDRangeKernel(clCxt->command_queue,kernel,2,NULL,globalthreads,
                                              localthreads,0,NULL,&time_event);
    if(ret != CL_SUCCESS){
        cout << "Failed to EnqueueNDRangeKernel. error code:"<< ret  << endl;
        return ;
    }
    clWaitForEvents(1,&time_event);
    clFinish(clCxt->command_queue);
    ret = clGetEventProfilingInfo(time_event,CL_PROFILING_COMMAND_QUEUED, sizeof(cl_ulong) ,&queued,NULL);
    if(ret != CL_SUCCESS){
        cout << "Failed to clGetEventProfilingInfo 1. error code:"<< ret  << endl;
    }
    ret = clGetEventProfilingInfo(time_event,CL_PROFILING_COMMAND_START, sizeof(cl_ulong) ,&start,NULL);
    if(ret != CL_SUCCESS){
        cout << "Failed to clGetEventProfilingInfo 2. error code:"<< ret  << endl;
    }
    ret = clGetEventProfilingInfo(time_event,CL_PROFILING_COMMAND_END, sizeof(cl_ulong) ,&end,NULL);
    if(ret != CL_SUCCESS){
        cout << "Failed to clGetEventProfilingInfo 3. error code:"<< ret  << endl;
    }
    cl_ulong kernelRealExecTimeNs = end - start;
    cl_ulong kernelTotalExecTimeNs = end - queued;
    if(ret == CL_SUCCESS)
        timeRcd.totaltime += (double)kernelTotalExecTimeNs/1000000;
    else
        timeRcd.totaltime = 0;   
    if(ret == CL_SUCCESS)
        timeRcd.kerneltime += (double)kernelRealExecTimeNs/1000000;
    else
        timeRcd.kerneltime = 0; 
    cout<<"KernelName:"<<left<<setw(10)<<kernelName; 
    printf("kernel  time=%lf  ",(double)kernelRealExecTimeNs/1000000);
    printf("total  time=%lf  \n",(double)kernelTotalExecTimeNs/1000000);
    ret = clReleaseKernel(kernel);
    if(ret != CL_SUCCESS)
        cout << "Failed to release kernel. error code:"<< ret  << endl;
    free(source_str);
    return ;
}

void create(clContext *clCxt, cl_mem *mem, int len)
{
    cl_int ret;
    *mem = clCreateBuffer(clCxt->context,CL_MEM_READ_WRITE,len,NULL,&ret);
    if(ret != CL_SUCCESS){
        cout << "Failed to create buffer on GPU." << endl;
        return ;
    }
}

void upload(clContext *clCxt,void *data,cl_mem *gdata,int datalen)
{
    //write data to buffer
    cl_int ret;
    ret = clEnqueueWriteBuffer(clCxt->command_queue,*gdata,CL_TRUE,0,datalen,(void *)data,0,NULL,NULL);
    if(ret != CL_SUCCESS){
        cout << "clEnqueueWriteBuffer failed." << endl;
        return ;
    }
    clFinish(clCxt->command_queue);
}

void download(clContext *clCxt,cl_mem *gdata,void *data,int data_len)
{
    cl_int ret;
    ret = clEnqueueReadBuffer(clCxt->command_queue, *gdata, CL_TRUE, 0, data_len,(void *)data, 0, NULL,NULL);
    if(ret != CL_SUCCESS){
        cout << "clEnqueueReadBuffer failed." << endl;
        return ;
    }
    clFinish(clCxt->command_queue);
}

void arrayinit(int *data,int elemnum,int range)
{
    for(int i = 0;i< elemnum;i++)
    {
       data[i] = rand() % range; 
    }
    return;
}
