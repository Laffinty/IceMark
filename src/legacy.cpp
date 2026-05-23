#include <windows.h>
#include <iostream>
#include <fstream>
#include <time.h>
#include <process.h>
#include <string>
#include <winbase.h>
using namespace std;

DWORD WINAPI Fun1Proc(
  LPVOID lpParameter   // thread data
);
DWORD WINAPI Fun2Proc(
  LPVOID lpParameter   // thread data
);
DWORD WINAPI Fun3Proc(
  LPVOID lpParameter   // thread data
);
DWORD WINAPI Fun4Proc(
  LPVOID lpParameter   // thread data
);
DWORD WINAPI Fun5Proc(
  LPVOID lpParameter   // thread data
);
DWORD WINAPI Fun6Proc(
  LPVOID lpParameter   // thread data
);
DWORD WINAPI Fun7Proc(
  LPVOID lpParameter   // thread data
);

int apple01=0;
double apple02,apple03,apple04,apple05;
HANDLE hMutex;

void main()
{
	cout<<"IceMark BETA"<<endl;
	cout<<"Nagi-Project 作者：leeways"<<endl;
	system("pause");
	HANDLE hThread1;
	HANDLE hThread2;
	HANDLE hThread3;
	HANDLE hThread7;
	hThread1=CreateThread(NULL,0,Fun1Proc,NULL,0,NULL);
	hThread2=CreateThread(NULL,0,Fun2Proc,NULL,0,NULL);
	hThread3=CreateThread(NULL,0,Fun3Proc,NULL,0,NULL);
	hThread7=CreateThread(NULL,0,Fun7Proc,NULL,0,NULL);

cout<<endl<<"多线程测试A 启动"<<endl;
	for(int a=50;apple01<4;a++)
{
	a--;
}
double x1=(1000000/apple02+1000000/apple03+1000000/apple04+1000000/apple05)/4;

apple01=0;
cout<<"多线程测试A Ok"<<endl;
CloseHandle(hThread1);
CloseHandle(hThread2);
CloseHandle(hThread3);
CloseHandle(hThread7);

HANDLE hThread4;
HANDLE hThread5;
hThread4=CreateThread(NULL,0,Fun4Proc,NULL,0,NULL);
hThread5=CreateThread(NULL,0,Fun5Proc,NULL,0,NULL);
cout<<endl<<"多线程测试B 启动"<<endl;
for(int a=50;apple01<2;a++)
{
	a--;
}
double x2=(1000000/apple02+1000000/apple03)/2;

	apple01=0;
	cout<<endl<<"多线程测试B Ok"<<endl;
CloseHandle(hThread4);
CloseHandle(hThread5);

HANDLE hThread6;
hThread6=CreateThread(NULL,0,Fun6Proc,NULL,0,NULL);

	cout<<endl<<"单线程测试 启动"<<endl;
for(int a=50;apple01<1;a++)
{
	a--;
}
double x3=1000000/apple02;



CloseHandle(hThread6);
cout<<"单线程测试 Ok"<<endl;

double x4=x1*20+x2*40+x3*30;
cout<<"========================"<<endl;
cout<<"最终得分："<<x4<<"分"<<endl;
cout<<"========================"<<endl;
	system("pause");
}

DWORD WINAPI Fun1Proc(LPVOID lpParameter)
{
	WaitForSingleObject(hMutex,INFINITE);
	BOOL WINAPI SetThreadPriority(HANDLE hThread1,int read1=2);
	clock_t op,ed;
	op=clock();
	long int n=10000;
	int *a;
	a = new int [n+10];
	if (a==NULL)
	{
		cout <<"分配内存空间失败!\n";exit(1);
	}
	long int m=5*n;
	a[0]=2;
	long int m1=2*m+1,n1=2*m;int q;
	for (int j=1;j<n+10;j++)
	{
		n1=n1*10;q=n1%m1;
		a[j]=(n1-q)/m1;n1=q;
	}
	for (int i=m-1;i>=1;i--)
	{
		a[0]=a[0]+2;
		for (int j1=0;j1<n+9;j1++)
			a[j1]=a[j1]*i;
		int p;
		for (int j2=0;j2<n+9;j2++)
		{
			p=0;
			p=a[j2]%(2*i+1);
			a[j2]=(a[j2]-p)/(2*i+1);
			a[j2+1]=a[j2+1]+10*p;
		}
	}
	a[0]=a[0]+2;
	int p1;
	for (int j3=n+10-2;j3>=0;j3--)
		if (a[j3]>=10)
		{
			p1=a[j3]%10;
			a[j3-1]=a[j3-1]+(a[j3]-p1)/10;
			a[j3]=p1;
		}
		cout<<""<<endl;
ed=clock();
double fen1 = (double)(ed - op) / CLOCKS_PER_SEC;
apple02=fen1*1000;
apple01++;
	return 0;
}

DWORD WINAPI Fun2Proc(LPVOID lpParameter)
{
	WaitForSingleObject(hMutex,INFINITE);
    BOOL WINAPI SetThreadPriority(HANDLE hThread2,int read1=2);
	clock_t op,ed;
	op=clock();
	long int n=10000;
	int *a;
	a = new int [n+10];
	if (a==NULL)
	{
		cout <<"分配内存空间失败!\n";exit(1);
	}
	long int m=5*n;
	a[0]=2;
	long int m1=2*m+1,n1=2*m;int q;
	for (int j=1;j<n+10;j++)
	{
		n1=n1*10;q=n1%m1;
		a[j]=(n1-q)/m1;n1=q;
	}
	for (int i=m-1;i>=1;i--)
	{
		a[0]=a[0]+2;
		for (int j1=0;j1<n+9;j1++)
			a[j1]=a[j1]*i;
		int p;
		for (int j2=0;j2<n+9;j2++)
		{
			p=0;
			p=a[j2]%(2*i+1);
			a[j2]=(a[j2]-p)/(2*i+1);
			a[j2+1]=a[j2+1]+10*p;
		}
	}
	a[0]=a[0]+2;
	int p1;
	for (int j3=n+10-2;j3>=0;j3--)
		if (a[j3]>=10)
		{
			p1=a[j3]%10;
			a[j3-1]=a[j3-1]+(a[j3]-p1)/10;
			a[j3]=p1;
		}
ed=clock();
double fen1 = (double)(ed - op) / CLOCKS_PER_SEC;
apple03=fen1*1000;
apple01++;
	return 0;
}
DWORD WINAPI Fun3Proc(LPVOID lpParameter)
{
	WaitForSingleObject(hMutex,INFINITE);
	BOOL WINAPI SetThreadPriority(HANDLE hThread3,int read1=2);
	clock_t op,ed;
	op=clock();
	long int n=10000;
	int *a;
	a = new int [n+10];
	if (a==NULL)
	{
		cout <<"分配内存空间失败!\n";exit(1);
	}
	long int m=5*n;
	a[0]=2;
	long int m1=2*m+1,n1=2*m;int q;
	for (int j=1;j<n+10;j++)
	{
		n1=n1*10;q=n1%m1;
		a[j]=(n1-q)/m1;n1=q;
	}
	for (int i=m-1;i>=1;i--)
	{
		a[0]=a[0]+2;
		for (int j1=0;j1<n+9;j1++)
			a[j1]=a[j1]*i;
		int p;
		for (int j2=0;j2<n+9;j2++)
		{
			p=0;
			p=a[j2]%(2*i+1);
			a[j2]=(a[j2]-p)/(2*i+1);
			a[j2+1]=a[j2+1]+10*p;
		}
	}
	a[0]=a[0]+2;
	int p1;
	for (int j3=n+10-2;j3>=0;j3--)
		if (a[j3]>=10)
		{
			p1=a[j3]%10;
			a[j3-1]=a[j3-1]+(a[j3]-p1)/10;
			a[j3]=p1;
		}
		ed=clock();
double fen1 = (double)(ed - op) / CLOCKS_PER_SEC;
apple04=fen1*1000;
apple01++;
	return 0;
}

DWORD WINAPI Fun4Proc(LPVOID lpParameter)
{
	WaitForSingleObject(hMutex,INFINITE);
	BOOL WINAPI SetThreadPriority(HANDLE hThread4,int read1=2);
	clock_t op,ed;
	op=clock();
	long int n=10000;
	int *a;
	a = new int [n+10];
	if (a==NULL)
	{
		cout <<"分配内存空间失败!\n";exit(1);
	}
	long int m=5*n;
	a[0]=2;
	long int m1=2*m+1,n1=2*m;int q;
	for (int j=1;j<n+10;j++)
	{
		n1=n1*10;q=n1%m1;
		a[j]=(n1-q)/m1;n1=q;
	}
	for (int i=m-1;i>=1;i--)
	{
		a[0]=a[0]+2;
		for (int j1=0;j1<n+9;j1++)
			a[j1]=a[j1]*i;
		int p;
		for (int j2=0;j2<n+9;j2++)
		{
			p=0;
			p=a[j2]%(2*i+1);
			a[j2]=(a[j2]-p)/(2*i+1);
			a[j2+1]=a[j2+1]+10*p;
		}
	}
	a[0]=a[0]+2;
	int p1;
	for (int j3=n+10-2;j3>=0;j3--)
		if (a[j3]>=10)
		{
			p1=a[j3]%10;
			a[j3-1]=a[j3-1]+(a[j3]-p1)/10;
			a[j3]=p1;
		}
		cout<<""<<endl;
ed=clock();
double fen1 = (double)(ed - op) / CLOCKS_PER_SEC;
apple02=fen1*1000;
apple01++;
	return 0;
}

DWORD WINAPI Fun5Proc(LPVOID lpParameter)
{
	WaitForSingleObject(hMutex,INFINITE);
	BOOL WINAPI SetThreadPriority(HANDLE hThread5,int read1=2);
	clock_t op,ed;
	op=clock();
	long int n=10000;
	int *a;
	a = new int [n+10];
	if (a==NULL)
	{
		cout <<"分配内存空间失败!\n";exit(1);
	}
	long int m=5*n;
	a[0]=2;
	long int m1=2*m+1,n1=2*m;int q;
	for (int j=1;j<n+10;j++)
	{
		n1=n1*10;q=n1%m1;
		a[j]=(n1-q)/m1;n1=q;
	}
	for (int i=m-1;i>=1;i--)
	{
		a[0]=a[0]+2;
		for (int j1=0;j1<n+9;j1++)
			a[j1]=a[j1]*i;
		int p;
		for (int j2=0;j2<n+9;j2++)
		{
			p=0;
			p=a[j2]%(2*i+1);
			a[j2]=(a[j2]-p)/(2*i+1);
			a[j2+1]=a[j2+1]+10*p;
		}
	}
	a[0]=a[0]+2;
	int p1;
	for (int j3=n+10-2;j3>=0;j3--)
		if (a[j3]>=10)
		{
			p1=a[j3]%10;
			a[j3-1]=a[j3-1]+(a[j3]-p1)/10;
			a[j3]=p1;
		}
ed=clock();
double fen1 = (double)(ed - op) / CLOCKS_PER_SEC;
apple03=fen1*1000;
apple01++;
	return 0;
}
DWORD WINAPI Fun6Proc(LPVOID lpParameter)
{
	WaitForSingleObject(hMutex,INFINITE);
BOOL WINAPI SetThreadPriority(HANDLE hThread6,int read1=2);
	clock_t op,ed;
	op=clock();
	long int n=10000;
	int *a;
	a = new int [n+10];
	if (a==NULL)
	{
		cout <<"分配内存空间失败!\n";exit(1);
	}
	long int m=5*n;
	a[0]=2;
	long int m1=2*m+1,n1=2*m;int q;
	for (int j=1;j<n+10;j++)
	{
		n1=n1*10;q=n1%m1;
		a[j]=(n1-q)/m1;n1=q;
	}
	for (int i=m-1;i>=1;i--)
	{
		a[0]=a[0]+2;
		for (int j1=0;j1<n+9;j1++)
			a[j1]=a[j1]*i;
		int p;
		for (int j2=0;j2<n+9;j2++)
		{
			p=0;
			p=a[j2]%(2*i+1);
			a[j2]=(a[j2]-p)/(2*i+1);
			a[j2+1]=a[j2+1]+10*p;
		}
	}
	a[0]=a[0]+2;
	int p1;
	for (int j3=n+10-2;j3>=0;j3--)
		if (a[j3]>=10)
		{
			p1=a[j3]%10;
			a[j3-1]=a[j3-1]+(a[j3]-p1)/10;
			a[j3]=p1;
		}
		ed=clock();
double fen1 = (double)(ed - op) / CLOCKS_PER_SEC;
apple04=fen1*1000;
apple01++;
	return 0;
}

DWORD WINAPI Fun7Proc(LPVOID lpParameter)
{
	WaitForSingleObject(hMutex,INFINITE);
	BOOL WINAPI SetThreadPriority(HANDLE hThread7,int read1=2);
	clock_t op,ed;
	op=clock();
	long int n=10000;
	int *a;
	a = new int [n+10];
	if (a==NULL)
	{
		cout <<"分配内存空间失败!\n";exit(1);
	}
	long int m=5*n;
	a[0]=2;
	long int m1=2*m+1,n1=2*m;int q;
	for (int j=1;j<n+10;j++)
	{
		n1=n1*10;q=n1%m1;
		a[j]=(n1-q)/m1;n1=q;
	}
	for (int i=m-1;i>=1;i--)
	{
		a[0]=a[0]+2;
		for (int j1=0;j1<n+9;j1++)
			a[j1]=a[j1]*i;
		int p;
		for (int j2=0;j2<n+9;j2++)
		{
			p=0;
			p=a[j2]%(2*i+1);
			a[j2]=(a[j2]-p)/(2*i+1);
			a[j2+1]=a[j2+1]+10*p;
		}
	}
	a[0]=a[0]+2;
	int p1;
	for (int j3=n+10-2;j3>=0;j3--)
		if (a[j3]>=10)
		{
			p1=a[j3]%10;
			a[j3-1]=a[j3-1]+(a[j3]-p1)/10;
			a[j3]=p1;
		}
		ed=clock();
double fen1 = (double)(ed - op) / CLOCKS_PER_SEC;
apple05=fen1*1000;
apple01++;
	return 0;
}