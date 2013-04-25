#include <stdio.h>
#include <sys/time.h>
#include <string.h>
#include <math.h>
#include <iostream>
#include <iomanip>
#include <string>
#include <limits>
#include <vector>
#include <stdlib.h>
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif
#define MAX_SOURCE_SIZE (0x100000)

using namespace std;

typedef struct clContext{
    cl_device_id device_id;
    cl_context context;
    cl_command_queue command_queue;
    cl_platform_id platform_id;
    cl_uint num_devices;
    cl_uint num_platforms;
    vector< pair<string,cl_program *> > program;   
}clContext;

typedef struct Plan{
    int localthread;
    int cta;
    int workgroup;
    int registergroup;
    int localmemgroup;
    int vectorlength;
    int coalesced;
    int dynamic_task;
}Plan;

typedef struct TimeRecord{
   double kerneltime,totaltime;
   double min_kerneltime, min_totaltime;
   double min_cputime;
}TimeRcd;

void getClContext(clContext *clCxt);
void releaseContext(clContext *clCxt);
void executeKernel(string source,string kernelName, vector< pair<size_t,const void *> > args,
                         const size_t * globalthreads,const size_t* localthreads,
                         char * build_options,clContext *clCxt);
void create(clContext *clCxt, cl_mem *mem, int len);
void upload(clContext *clCxt,void *data,cl_mem *gdata,int datalen);
void download(clContext *clCxt,cl_mem *gdata,void *data,int data_len);
void arrayinit(int *data,int elemnum,int range);
