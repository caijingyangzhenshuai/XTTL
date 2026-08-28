#include<stdio.h>
#include<stdlib.h>
int main()
{
	printf("******real-time info tools,written by kzzk heyuan!******\n");
	//printf("use: please su - root\n");
	printf("real-time power and temperature\n");
	printf("------------------\n");
	/*
	while(1)
	{
		system("ascend-dmi -i | grep Atlas | awk '{print $(NF-1)}'");
	}
	*/
	FILE *fp1;
	FILE *fp2;
	char pw[7];// pw is short formt of power
	char tp[7];// tp is short formt of temper
	while(1)
	{
		//fp = popen("ascend-dmi -i | grep Ascend | awk '{print $(NF-1)}'","r");
		fp1 = popen("npu-smi info | grep OK | head -n 1 | awk '{print $(NF-5)}'","r");
		fp2 = popen("npu-smi info | grep OK | head -n 1 | awk '{print $(NF-4)}'","r");
		fgets(pw,7,fp1);
		fgets(tp,7,fp2);
		pclose(fp1);
		pclose(fp2);
		//printf("%s",pw);
		printf("功耗:%s",pw);	
		printf("温度:%s",tp);
		printf("------------------\n");
		//puts(pw);
		//printf("温度:");
		//puts(tp);

	}
	return 0;
}
