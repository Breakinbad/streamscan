// @Authors
//    Shengen Yan,yanshengen@gmail.com
/**************************************PUBLICFUNC*************************************/
#pragma OPENCL EXTENSION cl_khr_fp64:enable

#define LOG_NUM_BANKS 5 
#define NUM_BANKS 32 
#define GET_CONFLICT_OFFSET(lid) ((lid) >> LOG_NUM_BANKS)
#define AVOID_CONFLICT(lid) ((lid) + ((lid) >> LOG_NUM_BANKS))
#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics:enable
#pragma OPENCL EXTENSION cl_khr_global_int32_extended_atomics:enable

#if defined (NB_L64)
#define NB_LSIZE 64
#define NB_LSIZE_LOG 6 
#define H_NB_LSIZE 32
#define NB_LSIZE_1 63
#define NB_LSIZE_2 65
#if defined (NB_CTA_8)
#define NB_CTA_NUM 8 
#define NB_CTA_SIZE 8
#define STEP_LOG 3 
#define H_NB_CTA_SIZE 4 
#define H_NB_CTA_SIZE_1 3
#define NB_CTA_SIZE_2 5
#define NB_CTA_SIZE_1 7
#define NB_CTA_LOG 3
#define NB_CTA_LOG_1 2
#endif
#if defined (NB_CTA_16)
#define NB_CTA_NUM 4 
#define NB_CTA_SIZE 16
#define STEP_LOG 4 
#define H_NB_CTA_SIZE 8 
#define H_NB_CTA_SIZE_1 7
#define NB_CTA_SIZE_2 17
#define NB_CTA_SIZE_1 15
#define NB_CTA_LOG 4
#define NB_CTA_LOG_1 3
#endif
#if defined (NB_CTA_32)
#define NB_CTA_NUM 2
#define NB_CTA_SIZE 32
#define STEP_LOG 5 
#define H_NB_CTA_SIZE 16 
#define H_NB_CTA_SIZE_1 15
#define NB_CTA_SIZE_2 33
#define NB_CTA_SIZE_1 31
#define NB_CTA_LOG 5
#define NB_CTA_LOG_1 4
#endif
#if defined (NB_CTA_64)
#define NB_CTA_NUM 1
#define NB_CTA_SIZE 64
#define STEP_LOG 6 
#define H_NB_CTA_SIZE 32 
#define H_NB_CTA_SIZE_1 31
#define NB_CTA_SIZE_2 65
#define NB_CTA_SIZE_1 63
#define NB_CTA_LOG 6
#define NB_CTA_LOG_1 5
#endif
#endif

#if defined (NB_L128)
#define NB_LSIZE 128
#define NB_LSIZE_LOG 7 
#define H_NB_LSIZE 64
#define NB_LSIZE_1 127
#define NB_LSIZE_2 129
#if defined (NB_CTA_8)
#define NB_CTA_NUM 16
#define NB_CTA_SIZE 8
#define STEP_LOG 3 
#define H_NB_CTA_SIZE 4 
#define H_NB_CTA_SIZE_1 3
#define NB_CTA_SIZE_2 5
#define NB_CTA_SIZE_1 7
#define NB_CTA_LOG 3
#define NB_CTA_LOG_1 2
#endif
#if defined (NB_CTA_16)
#define NB_CTA_NUM 8
#define NB_CTA_SIZE 16
#define STEP_LOG 4 
#define H_NB_CTA_SIZE 8 
#define H_NB_CTA_SIZE_1 7
#define NB_CTA_SIZE_2 17
#define NB_CTA_SIZE_1 15
#define NB_CTA_LOG 4
#define NB_CTA_LOG_1 3
#endif
#if defined (NB_CTA_32)
#define NB_CTA_NUM 4
#define NB_CTA_SIZE 32
#define STEP_LOG 5 
#define H_NB_CTA_SIZE 16 
#define H_NB_CTA_SIZE_1 15
#define NB_CTA_SIZE_2 33
#define NB_CTA_SIZE_1 31
#define NB_CTA_LOG 5
#define NB_CTA_LOG_1 4
#endif
#if defined (NB_CTA_64)
#define NB_CTA_NUM 2
#define NB_CTA_SIZE 64
#define STEP_LOG 6 
#define H_NB_CTA_SIZE 32 
#define H_NB_CTA_SIZE_1 31
#define NB_CTA_SIZE_2 65
#define NB_CTA_SIZE_1 63
#define NB_CTA_LOG 6
#define NB_CTA_LOG_1 5
#endif
#endif

#if defined (NB_L256)
#define NB_LSIZE 256 
#define NB_LSIZE_LOG 8 
#define H_NB_LSIZE 128
#define NB_LSIZE_1 255 
#define NB_LSIZE_2 257 
#if defined (NB_CTA_8)
#define NB_CTA_NUM 32 
#define NB_CTA_SIZE 8
#define STEP_LOG 3 
#define H_NB_CTA_SIZE 4 
#define H_NB_CTA_SIZE_1 3
#define NB_CTA_SIZE_2 5
#define NB_CTA_SIZE_1 7
#define NB_CTA_LOG 3
#define NB_CTA_LOG_1 2
#endif
#if defined (NB_CTA_16)
#define NB_CTA_NUM 16 
#define NB_CTA_SIZE 16
#define STEP_LOG 4 
#define H_NB_CTA_SIZE 8 
#define H_NB_CTA_SIZE_1 7
#define NB_CTA_SIZE_2 17
#define NB_CTA_SIZE_1 15
#define NB_CTA_LOG 4
#define NB_CTA_LOG_1 3
#endif
#if defined (NB_CTA_32)
#define NB_CTA_NUM 8
#define NB_CTA_SIZE 32
#define STEP_LOG 5
#define H_NB_CTA_SIZE 16 
#define H_NB_CTA_SIZE_1 15
#define NB_CTA_SIZE_2 33
#define NB_CTA_SIZE_1 31
#define NB_CTA_LOG 5
#define NB_CTA_LOG_1 4
#endif
#if defined (NB_CTA_64)
#define NB_CTA_NUM 4
#define NB_CTA_SIZE 64
#define STEP_LOG 6
#define H_NB_CTA_SIZE 32 
#define H_NB_CTA_SIZE_1 31
#define NB_CTA_SIZE_2 65
#define NB_CTA_SIZE_1 63
#define NB_CTA_LOG 6
#define NB_CTA_LOG_1 5
#endif
#endif
//#define DYNAMIC_TASK
#if defined (NB_VEC_1)
__kernel void scan(__global int *src,__global volatile int *sum, __global int *dst,int groupnum)
{
    unsigned int lid = get_local_id(0);
#if defined DYNAMIC_TASK
    unsigned int gid;
    __local int gid_;
    if(lid == 0)
        gid_ = atom_add((__global int*)(sum+groupnum),1);
    barrier(CLK_LOCAL_MEM_FENCE);
    gid = gid_;
#else
   unsigned int gid = get_group_id(0);
#endif
    __local int lm[NB_CTA_NUM][NB_CTA_SIZE][NB_LOCAL_SIZE];
    __local int column[2][NB_LSIZE_2],lpsum;
    int re[NB_REG_SIZE];
    int kgp = lid >> NB_CTA_LOG;
    int kid = lid & NB_CTA_SIZE_1;
    int lid_= lid+1;
    int src_id=gid * STEP_NUM * NB_LSIZE + kgp * NB_CTA_SIZE * STEP_NUM + kid;
    int t1=0,t2=0,t3=0,t4=0,psum=0,rsum=0;
#if NB_REG_GRP   
    for(t1=0;t1<NB_REG_GRP;t1++)
    {
        t3=NB_CTA_SIZE*t1+src_id;
        for(t2=0;t2<NB_CTA_SIZE;t2++)
        {
            lm[kgp][t2][kid]=src[t3 + t2*STEP_NUM];
        }
        t3=NB_CTA_SIZE*t1;
        re[t3]=lm[kgp][kid][0] + psum;
        for(t2=1;t2<NB_CTA_SIZE;t2++)
        {
            re[t2+t3]=re[t2-1+t3] + lm[kgp][kid][t2];
        }
        psum=re[t3+NB_CTA_SIZE_1];
        rsum=psum;
    }
#endif
    for(t1=0;t1<NB_LOCAL_GRP;t1++)
    {
        t3=NB_CTA_SIZE*(t1+NB_REG_GRP)+src_id;
        t4=NB_CTA_SIZE*t1;
        for(t2=0;t2<NB_CTA_SIZE;t2++)
        {
            lm[kgp][t2][t4+kid] = src[t3 + t2*STEP_NUM];
        }   
        for(t2=0;t2<NB_CTA_SIZE;t2++)
            psum +=lm[kgp][kid][t4+t2];
    }
    column[0][lid_] = psum;
     
    barrier(CLK_LOCAL_MEM_FENCE);
    for(t1 =1,t2 = 1,t3=1;t1<=H_NB_LSIZE; t1<<=1,t2 <<=1, t3 = t3^1){
        column[t3][lid_] = lid >=t1 ? column[t3^1][lid_] + column[t3^1][lid_ - t2] : column[t3^1][lid_];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    psum=0; 
    t3=t3^1;
     
    if(lid==0)
    {
        if(gid == 0)
            sum[0] = column[t3][NB_LSIZE];
        else
        {
            while((psum=sum[gid - 1])==0){}
            sum[gid] = column[t3][NB_LSIZE] + psum;
        }
        lpsum = psum;
        column[t3][0]=0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    psum=lpsum + column[t3][lid];
    lm[kgp][kid][0] = lm[kgp][kid][0] + psum + rsum;
    for(t1=1;t1<NB_LOCAL_SIZE;t1++)
        lm[kgp][kid][t1] += lm[kgp][kid][t1-1];
    for(t1=0;t1<NB_LOCAL_GRP;t1++)
    {
        t3=NB_CTA_SIZE*(t1+NB_REG_GRP)+src_id;
        t4=NB_CTA_SIZE*t1;
        for(t2=0;t2<NB_CTA_SIZE;t2++)
        {
            dst[t3 + t2*STEP_NUM]=lm[kgp][t2][t4+kid];
        }   
    }
#if NB_REG_GRP
    for(t1=0;t1<NB_REG_GRP;t1++)
    {
        t3=NB_CTA_SIZE*t1;
        for(t2=0;t2<NB_CTA_SIZE;t2++)
        {
            lm[kgp][kid][t2]=re[t2+t3] + psum;
        }
        t3=NB_CTA_SIZE*t1+src_id;
        for(t2=0;t2<NB_CTA_SIZE;t2++)
        {
            dst[t3 + t2*STEP_NUM]=lm[kgp][t2][kid];
        }
    }
#endif    
}
#elif defined (NB_VEC_2)
__kernel void scan(__global int2 *src,__global volatile int *sum, __global int2 *dst,int groupnum)
{
    unsigned int lid = get_local_id(0);
#if defined DYNAMIC_TASK
    unsigned int gid;
    __local int gid_;
    if(lid == 0)
        gid_ = atom_add((__global int*)(sum+groupnum),1);
    barrier(CLK_LOCAL_MEM_FENCE);
    gid = gid_;
#else
   unsigned int gid = get_group_id(0);
#endif
    __local int lm[NB_CTA_NUM][NB_CTA_SIZE][NB_LOCAL_SIZE];
    __local int column[2][NB_LSIZE_2],lpsum;
    int re[NB_REG_SIZE];
    int2 temp;
    int kgp = lid >> NB_CTA_LOG;
    int kid = lid & NB_CTA_SIZE_1;
    int lid_= lid+1;
    int kgp2= kid>>(NB_CTA_LOG_1);
    int kid2= lid & (H_NB_CTA_SIZE_1);
    int kid3= kid2*2;
    int kid4= kid3+1;
    int src_id=gid * STEP_NUM * NB_LSIZE + kgp * NB_CTA_SIZE * STEP_NUM + kgp2 * STEP_NUM + kid2;
    int t1=0,t2=0,t3=0,psum=0,rsum=0;
#if NB_REG_GRP   
    for(t1=0;t1<NB_REG_GRP;t1++)
    {
        t3=H_NB_CTA_SIZE*t1+src_id;
        for(t2=0;t2<NB_CTA_SIZE;t2+=2)
        {
            temp=src[t3 + t2*STEP_NUM];
            lm[kgp][t2+kgp2][kid3] = temp.s0;
            lm[kgp][t2+kgp2][kid4] = temp.s1;
        }
        t3=NB_CTA_SIZE*t1;
        re[t3]=lm[kgp][kid][0] + psum;
        for(t2=1;t2<NB_CTA_SIZE;t2++)
        {
            re[t2+t3]=re[t2-1+t3] + lm[kgp][kid][t2];
        }
        psum=re[t3+NB_CTA_SIZE_1];
        rsum=psum;
    }
#endif
    for(t1=0;t1<NB_LOCAL_GRP;t1++)
    {
        t3=H_NB_CTA_SIZE*(t1+NB_REG_GRP)+src_id;
        for(t2=0;t2<NB_CTA_SIZE;t2+=2)
        {
            temp=src[t3 + t2*STEP_NUM];
            lm[kgp][t2+kgp2][kid3+NB_CTA_SIZE*t1] = temp.s0;
            lm[kgp][t2+kgp2][kid4+NB_CTA_SIZE*t1] = temp.s1;
        }   
        t3=NB_CTA_SIZE*t1;
        for(t2=0;t2<NB_CTA_SIZE;t2++)
            psum +=lm[kgp][kid][t2+t3];
    }
    column[0][lid_] = psum;
     
    barrier(CLK_LOCAL_MEM_FENCE);
    for(t1 =1,t2 = 1,t3=1;t1<=H_NB_LSIZE; t1<<=1,t2 <<=1, t3 = t3^1){
        column[t3][lid_] = lid >=t1 ? column[t3^1][lid_] + column[t3^1][lid_ - t2] : column[t3^1][lid_];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    psum=0; 
    t3=t3^1;
    
    if(lid==0)
    {
        if(gid == 0)
            sum[0] = column[t3][NB_LSIZE];
        else
        {
            while((psum=sum[gid - 1])==0){}
            sum[gid] = column[t3][NB_LSIZE] + psum;
        }
        lpsum = psum;
        column[t3][0]=0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    psum=lpsum + column[t3][lid];
    lm[kgp][kid][0] = lm[kgp][kid][0] + psum + rsum;
    for(t1=1;t1<NB_LOCAL_SIZE;t1++)
        lm[kgp][kid][t1] += lm[kgp][kid][t1-1];
    for(t1=0;t1<NB_LOCAL_GRP;t1++)
    {
        t3=H_NB_CTA_SIZE*(t1+NB_REG_GRP)+src_id;
        for(t2=0;t2<NB_CTA_SIZE;t2+=2)
        {
            temp.s0 = lm[kgp][t2+kgp2][kid3+NB_CTA_SIZE*t1];
            temp.s1 = lm[kgp][t2+kgp2][kid4+NB_CTA_SIZE*t1];
            dst[t3 + t2*STEP_NUM]=temp;
        }  
    }
#if NB_REG_GRP
    for(t1=0;t1<NB_REG_GRP;t1++)
    {
        t3=NB_CTA_SIZE*t1;
        for(t2=0;t2<NB_CTA_SIZE;t2++)
        {
            lm[kgp][kid][t2]=re[t2+t3] + psum;
        }
        t3=H_NB_CTA_SIZE*t1+src_id;
        for(t2=0;t2<NB_CTA_SIZE;t2+=2)
        {
            temp.s0 = lm[kgp][t2+kgp2][kid3];
            temp.s1 = lm[kgp][t2+kgp2][kid4];
            dst[t3 + t2*STEP_NUM]=temp;
        }
    }
#endif    
}
#endif 

__kernel void scantail(__global int *src,__global volatile int *sum, __global int *dst,int elemnum,int pregroupnum,int groupnum,int len)
{
    unsigned int lid = get_local_id(0);
#if defined DYNAMIC_TASK
    unsigned int gid;
    __local int gid_;
    if(lid == 0)
        gid_ = atom_add((__global int*)(sum+groupnum+pregroupnum+1),1);
    barrier(CLK_LOCAL_MEM_FENCE);
    gid = gid_;
#else
   unsigned int gid = get_group_id(0);
#endif
    __local int lm[NB_CTA_NUM][NB_CTA_SIZE][NB_LOCAL_SIZE];
    __local int column[2][NB_LSIZE_2],lpsum;
    int re[NB_REG_SIZE];
    int kgp = lid >> NB_CTA_LOG;
    int kid = lid & NB_CTA_SIZE_1;
    int lid_= lid+1;
    int src_id=gid * STEP_NUM * NB_LSIZE + kgp * NB_CTA_SIZE * STEP_NUM + kid + elemnum-len;
    int t1=0,t2=0,t3=0,t4=0,psum=0,rsum=0;
#if NB_REG_GRP   
    for(t1=0;t1<NB_REG_GRP;t1++)
    {
        t3=NB_CTA_SIZE*t1+src_id;
        for(t2=0;t2<NB_CTA_SIZE;t2++)
        {
            if(t3 + t2 * STEP_NUM < elemnum)
                lm[kgp][t2][kid]=src[t3 + t2*STEP_NUM];
            else
                lm[kgp][t2][kid]=0;
        }
        t3=NB_CTA_SIZE*t1;
        re[t3]=lm[kgp][kid][0] + psum;
        for(t2=1;t2<NB_CTA_SIZE;t2++)
        {
            re[t2+t3]=re[t2-1+t3] + lm[kgp][kid][t2];
        }
        psum=re[t3+NB_CTA_SIZE_1];
        rsum=psum;
    }
#endif
    for(t1=0;t1<NB_LOCAL_GRP;t1++)
    {
        t3=NB_CTA_SIZE*(t1+NB_REG_GRP)+src_id;
        t4=NB_CTA_SIZE*t1;
        for(t2=0;t2<NB_CTA_SIZE;t2++)
        {
            if(t3 + t2 * STEP_NUM < elemnum)
                lm[kgp][t2][t4+kid] = src[t3 + t2*STEP_NUM];
            else
                lm[kgp][t2][t4+kid] = 0;
        }   
        for(t2=0;t2<NB_CTA_SIZE;t2++)
            psum +=lm[kgp][kid][t4+t2];
    }
    column[0][lid_] = psum;
     
    barrier(CLK_LOCAL_MEM_FENCE);
    for(t1 =1,t2 = 1,t3=1;t1<=H_NB_LSIZE; t1<<=1,t2 <<=1, t3 = t3^1){
        column[t3][lid_] = lid >=t1 ? column[t3^1][lid_] + column[t3^1][lid_ - t2] : column[t3^1][lid_];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    psum=0; 
    t3=t3^1;
     
    if(lid==0)
    {
        if(gid == 0){
            if(pregroupnum!=0)
                psum = sum[pregroupnum-1];
            else
                psum = 0;
            sum[pregroupnum+1] = psum + column[t3][NB_LSIZE];
        }
        else
        {
            while((psum=sum[pregroupnum + gid])==0){}
            sum[pregroupnum + 1 + gid] = column[t3][NB_LSIZE] + psum;
        }
        lpsum = psum;
        column[t3][0]=0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    psum=lpsum + column[t3][lid];
    lm[kgp][kid][0] = lm[kgp][kid][0] + psum + rsum;
    for(t1=1;t1<NB_LOCAL_SIZE;t1++)
        lm[kgp][kid][t1] += lm[kgp][kid][t1-1];
    for(t1=0;t1<NB_LOCAL_GRP;t1++)
    {
        t3=NB_CTA_SIZE*(t1+NB_REG_GRP)+src_id;
        t4=NB_CTA_SIZE*t1;
        for(t2=0;t2<NB_CTA_SIZE;t2++)
        {
            if(t3 + t2 * STEP_NUM < elemnum)
                dst[t3 + t2*STEP_NUM]=lm[kgp][t2][t4+kid];
        }   
    }
#if NB_REG_GRP
    for(t1=0;t1<NB_REG_GRP;t1++)
    {
        t3=NB_CTA_SIZE*t1;
        for(t2=0;t2<NB_CTA_SIZE;t2++)
        {
            lm[kgp][kid][t2]=re[t2+t3] + psum;
        }
        t3=NB_CTA_SIZE*t1+src_id;
        for(t2=0;t2<NB_CTA_SIZE;t2++)
        {
            if(t3 + t2 * STEP_NUM < elemnum)
                dst[t3 + t2*STEP_NUM]=lm[kgp][t2][kid];
        }
    }
#endif    
}
