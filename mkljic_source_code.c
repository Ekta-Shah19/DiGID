#include<stdio.h>
#include<string.h>
#include<stdlib.h>
#include<math.h>
#include<time.h>
#include<sys/timeb.h>
#include<sys/stat.h>

void compute_final_eigenvector(double **,double **,double ***,double *,double,int,int,int,int,int,char *);
void compute_individual_eigenvector(double **,double **,double **,double **,int,int,int,int,int,char *);
void compute_final_network(double **,double **,double **,int,int,int,int,double,double,char *);
void compute_individual_eigenvector1(double **,double **,double **,int,int,int,int,int,char *);
void update_unified_network(double **,double **,double ***,double *,double,int,int,int);
double compute_diff_trace(double **,double **,int,int);
double compute_trace3D(double ***,double ***,double *,int,int,int);
void compute_eigenvalue(double **,double **,double *,int,int);
void update_weight(double **,double ***,double **,double ***,double *,int,int,int,int);
void compute_eigenvector(char *,char *,char *,int,char *);
double compute_trace2D(double **,double **,int,int);
void compute_laplacian(double **,double **,int,int);
double compute_l1norm(double **,double **,int,int);
void compute_knn_graph(double **,int,int,int);
void write_to_file(char *,double **,int,int);
void createRscript_GS(char *,char *,char *);
void compute_inverse(char *,char *,char *);
void threshold_network(double **,int,int);
int checkifsymmetric(double **,int,int);
void deallocate3D(double ***,int,int);
int count_negative(double **,int,int);
void symmetricize(double **,int,int);
double ***allocate3D(int,int,int);
void normalize(double **,int,int);
int stop_check(double *,double *);
void deallocate2D(double **,int);
double **allocate2D(int,int);

int main(int argc,char *argv[])
{
	int i,j,k,n;
	int row,col;
	int iteration;
	int no_of_clusters;
	int nnz,negative;
	int temp,count;
	int no_of_neighbors;
	int flag;
	double itr_time,init_time,total_time;
	double alpha,beta,gamma;
	double value,objective;
	double *omega;
	double *eigenval,*eigenval1;
	double **eigenvec;
	double ***individual_eigenvec;
	double **unified,diff_final;
	double **final,**updated_final;
	double **laplacian,**final_laplacian;
	double ***network,***inverse;
	char comand[2000],fname[1000];
	char lap_file[1000];
	char eigvec_file[1000],eigval_file[1000];
	char eigvec[1000],ind_eigvec[1000];
	char omega_file[1000];
	char dir[500];
	time_t t;
	clock_t begin,end;
    struct timeb ti,tf;
    struct stat st;
	FILE *fp;
	
	n=atoi(argv[1]);
	no_of_clusters=atoi(argv[n+3]);
	alpha=atof(argv[n+4]);
	beta=atof(argv[n+5]);
	gamma=atof(argv[n+6]);
	
	sprintf(dir,"%s/%.1lfa_%.2lfb_%0.2lfg_%dc",argv[n+2],alpha,beta,gamma,no_of_clusters);
	sprintf(comand,"mkdir %s",dir);
	system(comand);
	
	/*-----------------------------------------------------------------------------------------*/
	//reading individual affinity networks and normalizing and symmetricizing each network.
	/*-----------------------------------------------------------------------------------------*/
	sprintf(comand,"wc -l %s > temp3.txt",argv[2]);
	system(comand);
	
	fp=fopen("temp3.txt","r");
	fscanf(fp,"%d",&row);
	fclose(fp);
	
	no_of_neighbors=(int)sqrt(row);
	
	begin=clock();
	col=row;
	network=allocate3D(n,row,col);
	omega=(double *)calloc(n,sizeof(double));
	for(i=0;i<n;i++)
	{
		fp=fopen(argv[2+i],"r");
		for(j=0;j<row;j++)
		{
			for(k=0;k<col;k++)
				fscanf(fp,"%lf",&network[i][j][k]);
			network[i][j][j]=0.0;
		}
		fclose(fp);
		
		omega[i]=1/(double)n;
		normalize(network[i],row,col);
		symmetricize(network[i],row,col);
		
		//sprintf(fname,"%s/NETWORK%d.txt",dir,i+1);
		//write_to_file(fname,network[i],row,col);
	}
	
	eigenval=(double *)calloc(no_of_clusters,sizeof(double));
	eigenval1=(double *)calloc(no_of_clusters,sizeof(double));
	unified=allocate2D(row,col);
	eigenvec=allocate2D(row,no_of_clusters);
	final=allocate2D(row,col);
	updated_final=allocate2D(row,col);
	final_laplacian=allocate2D(row,col);
	individual_eigenvec=allocate3D(n,row,no_of_clusters);
	inverse=allocate3D(n,row,col);
	
	
		iteration=0;
		sprintf(fname,"%s/FINAL_EIGENVECTOR_ITR%d.txt",dir,iteration);
		/*------------------------------------------------*/
		//creating unified network
		/*------------------------------------------------*/
		for(j=0;j<row;j++)
			for(k=0;k<col;k++)
				for(i=0;i<n;i++)
					unified[j][k]+=(omega[i]*network[i][j][k]);
		normalize(unified,row,col);
		symmetricize(unified,row,col);
	
		/*Since we are dealing with (D^(-1/2))*unified*(D^(-1/2)), so we need
		to compute the eigenvectors corresponding to the k largest eigenvalues.*/
		sprintf(eigvec_file,"%s/FINAL_EIGENVECTOR_ITR0.txt",dir);
		sprintf(eigval_file,"%s/FINAL_EIGENVALUE_ITR0.txt",dir);
		if(stat(eigvec_file,&st)!=0)
		{
			/*------------------------------------------------------------------------------------------------------------*/
			//Initially computing laplacian of "unified" network, writing to file and computing its largest eigenvectors
			/*------------------------------------------------------------------------------------------------------------*/
			laplacian=allocate2D(row,col);
			compute_laplacian(laplacian,unified,row,col);	//compute (D^(-1/2))*unified*(D^(-1/2)) in this part of the code
			sprintf(lap_file,"%s/LAPLACIAN.txt",dir);
			write_to_file(lap_file,laplacian,row,col);
			deallocate2D(laplacian,row);
			
			sprintf(eigvec,"compute_eigenvector.R");
			compute_eigenvector(lap_file,eigvec_file,eigval_file,no_of_clusters,eigvec);
			system("R CMD BATCH compute_eigenvector.R");
			system("cat compute_eigenvector.Rout");
			
			sprintf(comand,"rm %s", lap_file);
			system(comand);
		}
	
		fp=fopen(eigvec_file,"r");
		for(i=0;i<row;i++)
			for(j=0;j<no_of_clusters;j++)
				fscanf(fp,"%lf",&eigenvec[i][j]);
		fclose(fp);
		
		fp=fopen(eigval_file,"r");
		for(i=0;i<no_of_clusters;i++)
			fscanf(fp,"%lf\t%d\n",&eigenval[i],&temp);
		fclose(fp);
	
		/*------------------------------------------------*/
		//Computing the block diagonal matrix, "final"
		/*------------------------------------------------*/
		compute_final_network(final,unified,eigenvec,row,col,no_of_clusters,1,alpha,beta,dir);
		compute_knn_graph(final,row,col,no_of_neighbors);
		normalize(final,row,col);
		symmetricize(final,row,col);
	
		/*------------------------------------------------*/
		//Iterating over "final","unified" and "omega"
		/*------------------------------------------------*/
		sprintf(omega_file,"%s/OMEGA.txt",dir);
		fp=fopen(omega_file,"w");
		for(i=0;i<n;i++)
			fprintf(fp,"%lf\t",omega[i]);
		fprintf(fp,"\n");
		fclose(fp);
		
		diff_final=1.0;
	
	while((count<10)&&(iteration<100))
	{
		printf("***********************Iteration %d***********************\n",++iteration);
		begin=clock();
		/*------------------------------------------*/
		//Updating individual eigenvectors
		/*------------------------------------------*/
		for(i=0;i<n;i++)
			if(iteration==1)
			{
				laplacian=allocate2D(row,col);
				compute_laplacian(laplacian,network[i],row,col);
				compute_individual_eigenvector(individual_eigenvec[i],inverse[i],laplacian,eigenvec,i+1,row,col,no_of_clusters,iteration,dir);
				deallocate2D(laplacian,row);
			}
			else
				compute_individual_eigenvector1(individual_eigenvec[i],inverse[i],eigenvec,row,col,no_of_clusters,i+1,iteration,dir);
		
		if(iteration<21)
		{
			update_weight(unified,network,eigenvec,individual_eigenvec,omega,n,row,col,no_of_clusters);		//updating weights of individual network
			fp=fopen(omega_file,"a");
			for(i=0;i<n;i++)
				fprintf(fp,"%lf\t",omega[i]);
			fprintf(fp,"\n");
			fclose(fp);
		}
		
		/*------------------------------------------------*/
		//Updating the unified network "H"
		/*------------------------------------------------*/
		update_unified_network(unified,final,network,omega,gamma,row,col,n);		//updating the unified network using the weights, individual networks and S
		normalize(unified,row,col);
		symmetricize(unified,row,col);
		
		/*------------------------------------------------------------------*/
		//Updating final eigenvectors and approximating eigenvalues
		/*------------------------------------------------------------------*/
		value=beta/(double)gamma;
		compute_laplacian(final_laplacian,final,row,col);
		compute_final_eigenvector(eigenvec,final_laplacian,individual_eigenvec,omega,value,n,row,col,no_of_clusters,iteration,dir);
		compute_eigenvalue(final_laplacian,eigenvec,eigenval1,row,no_of_clusters);
		sprintf(fname,"%s/Eigenvalues.txt",dir);
		fp=fopen(fname,"a");
		fprintf(fp,"%d\t",iteration);
		for(i=0;i<no_of_clusters;i++)
			fprintf(fp,"%lf\t",eigenval1[i]);
		fprintf(fp,"\n");
		fclose(fp);
		
		/*------------------------------------------------*/
		//Computing the block diagonal matrix, "final"
		/*------------------------------------------------*/
		compute_final_network(updated_final,unified,eigenvec,row,col,no_of_clusters,iteration,alpha,beta,dir);		//learning block diagonal matrix, "final"
		compute_knn_graph(updated_final,row,col,no_of_neighbors);
		normalize(updated_final,row,col);
		symmetricize(updated_final,row,col);
		
		/*----------------------------------------------------------*/
		//Computing the l1-norm between "final(t)" and final(t+1)
		/*----------------------------------------------------------*/
		diff_final=compute_l1norm(final,updated_final,row,col);
		sprintf(fname,"%s/DIFFERENCE.txt",dir);
		fp=fopen(fname,"a");
		fprintf(fp,"%lf\n",diff_final);
		fclose(fp);
		
		for(i=0;i<row;i++)
			for(j=0;j<col;j++)
				final[i][j]=updated_final[i][j];
		
		if((iteration%2)==0)
		{
			flag=stop_check(eigenval,eigenval1);
			for(i=0;i<10;i++)
				eigenval[i]=eigenval1[i];
			if(flag==0)
				count=0;
			if(flag==1)
				count++;
		}
		end=clock();
		
		itr_time=(double)(end-begin)/CLOCKS_PER_SEC;
		printf("\nITERATION=%d\tTIME REQUIRED=%lf secs\n",iteration,itr_time);
		sprintf(fname,"%s/Time.txt",dir);
		fp=fopen(fname,"a");
		fprintf(fp,"ITERATION=%d\tTIME REQUIRED=%lf sec\n",iteration,itr_time);
		fclose(fp);
	}
	
	sprintf(fname,"%s/Time.txt",dir);
	fp=fopen(fname,"a");
	fprintf(fp,"ITERATION=%d\tTIME REQUIRED=%lf millisec\n",iteration,init_time);
	fclose(fp);
	
	sprintf(fname,"%s/UNIFIED_NETWORK.txt",dir);
	write_to_file(fname,unified,row,col);
			
	sprintf(fname,"%s/FINAL_NETWORK.txt",dir);
	write_to_file(fname,final,row,col);
	
	free(omega);
	free(eigenval);
	free(eigenval1);
	deallocate3D(network,n,row);
	deallocate2D(unified,row);
	deallocate2D(final,row);
	deallocate2D(eigenvec,row);
	deallocate2D(updated_final,row);
	deallocate2D(final_laplacian,row);
	deallocate3D(individual_eigenvec,n,row);
	deallocate3D(inverse,n,row);
	
	return 0;
}

int stop_check(double *eigenval,double *eigenval1)
{
	int i;
	double sum;
	
	sum=0.0;
	for(i=0;i<10;i++)
		sum+=fabs(eigenval[i]-eigenval1[i]);
	
	if(sum<0.0001)
		return 1;
	else
		return 0;
}

void write_to_file(char *fname,double **mat,int row,int col)
{
	int i,j;
	FILE *fp;
	
	fp=fopen(fname,"w");
	for(i=0;i<row;i++)
	{
		for(j=0;j<col-1;j++)
			fprintf(fp,"%.15lf\t",mat[i][j]);
		fprintf(fp,"%.15lf\n",mat[i][j]);
	}
	fclose(fp);
	
	return;
}

double compute_diff_trace(double **final_eigenvec,double **indiv_eigenvec,int row,int no_of_clusters)
{
	int i,j,k;
	double trace;
	double **temp;
	FILE *fp;
	
	trace=0.0;
	temp=allocate2D(row,no_of_clusters);
	for(i=0;i<row;i++)
		for(j=0;j<no_of_clusters;j++)
			temp[i][j]=final_eigenvec[i][j]-indiv_eigenvec[i][j];
	
	for(i=0;i<no_of_clusters;i++)
		for(j=0;j<row;j++)
			trace+=(temp[j][i]*temp[j][i]);
	deallocate2D(temp,row);
	
	return trace;
}

double compute_trace2D(double **final,double **eigenvec,int row,int no_of_clusters)
{
	int i,j,k;
	double trace;
	double **temp;
	FILE *fp;
	
	trace=0.0;
	temp=allocate2D(row,row);
	for(i=0;i<row;i++)
	{
		for(j=0;j<row;j++)
		{
			for(k=0;k<no_of_clusters;k++)
				temp[i][j]+=pow((eigenvec[i][k]-eigenvec[j][k]),2);
			trace+=(temp[i][j]*final[i][j]);
		}
	}
	deallocate2D(temp,row);
	
	return trace;
}

double compute_trace3D(double ***network,double ***vector,double *omega,int n,int row,int no_of_clusters)
{
	int i,j,k,l;
	double trace;
	double temp1;
	double **temp;
	FILE *fp;
	
	trace=0.0;
	for(l=0;l<n;l++)
	{
		temp1=0.0;
		temp=allocate2D(row,row);
		for(i=0;i<row;i++)
		{
			for(j=0;j<row;j++)
			{
				for(k=0;k<no_of_clusters;k++)
					temp[i][j]+=pow((vector[l][i][k]-vector[l][j][k]),2);
				temp1+=(temp[i][j]*network[l][i][j]);
			}
		}
		deallocate2D(temp,row);
		trace+=(omega[l]*temp1);
	}
	
	return trace;
}

void compute_eigenvalue(double **mat,double **vec,double *eigenvalue,int row,int no_of_clusters)
{
	int i,j,k;
	double *temp;
	
	for(i=0;i<no_of_clusters;i++)
	{
		eigenvalue[i]=0.0;
		temp=(double *)calloc(row,sizeof(double));
		for(j=0;j<row;j++)
			for(k=0;k<row;k++)
				temp[j]+=(vec[k][i]*mat[k][j]);
		
		for(j=0;j<row;j++)
			eigenvalue[i]+=(temp[j]*vec[j][i]);
		free(temp);
	}
	
	return;
}

void compute_knn_graph(double **mat,int row,int col,int no_of_neighbors)
{
	int i,j,k;
	int index;
	int *flag;
	
	for(i=0;i<row;i++)
	{
		flag=(int *)calloc(col,sizeof(int));
		for(j=0;j<no_of_neighbors;j++)
		{
			for(k=0;k<col;k++)
				if(flag[k]==0)
				{
					index=k;
					break;
				}
			
			for(k=0;k<col;k++)
				if((flag[k]==0)&&(mat[i][k]>mat[i][index]))
					index=k;
			flag[index]=1;
		}
		
		for(k=0;k<col;k++)
			if(flag[k]==0)
				mat[i][k]=0.0;
		free(flag);
	}
	return;
}

int count_negative(double **mat,int row,int col)
{
	int i,j;
	int count;
	
	count=0;
	for(i=0;i<row;i++)
		for(j=0;j<col;j++)
			if(mat[i][j]<0.0)
				count++;
	
	return count;
}

void threshold_network(double **mat,int row,int col)
{
	int i,j;
	double sum;
	
	sum=0.0;
	for(i=0;i<row;i++)
		for(j=0;j<col;j++)
			sum+=mat[i][j];
	sum=sum/(double)(row*col);
	
	for(i=0;i<row;i++)
		for(j=0;j<col;j++)
			if(mat[i][j]<sum)
				mat[i][j]=0.0;
	
	return;
}

void compute_laplacian(double **laplacian,double **network,int row,int col)
{
	int i,j;
	double *D;
	
	D=(double *)calloc(row,sizeof(double));
	for(i=0;i<row;i++)
		for(j=0;j<col;j++)
			D[i]+=network[i][j];
	
	for(i=0;i<row;i++)
		for(j=0;j<row;j++)
			if(i!=j)
			{
				if((D[i]>0.0)&&(D[j]>0.0))
					laplacian[i][j]=(-1)*(network[i][j]/(double)(sqrt(D[i]*D[j])));
				else
					laplacian[i][j]=0.0;
			}
			else
				laplacian[i][j]=1.0;
	
	free(D);
	
	return;
}

void compute_final_network(double **network,double **unified,double **eigenvec,int row,int col,int no_of_clusters,int iter,double alpha,double beta,char *path)
{
	int i,j,k;
	int flag;
	int nzcount;
	double **diff;
	char fname[1000];
	FILE *fp;
	
	nzcount=0;
	diff=allocate2D(row,col);
	for(i=0;i<row;i++)
	{
		for(j=0;j<col;j++)
		{
			diff[i][j]=0.0;
			for(k=0;k<no_of_clusters;k++)
				diff[i][j]+=(pow((eigenvec[i][k]-eigenvec[j][k]),2));
			network[i][j]=(unified[i][j]-(beta*0.5*diff[i][j]))/(double)(1+alpha);
			
			if(network[i][j]<0.0)
				network[i][j]=0.0;
			
			if(network[i][j]>0.0)
				nzcount++;
		}
	}
	
	deallocate2D(diff,row);
	
	return;
}

int checkifsymmetric(double **mat,int row,int col)
{
	int i,j;
	int flag;
	
	flag=1;
	for(i=0;i<row;i++)
		for(j=i;j<col;j++)
			if(fabs(mat[i][j]-mat[j][i])>1e-10)
			{
				flag=0;
				break;
			}
	
	return flag;
}

void update_unified_network(double **unified,double **final,double ***network,double *omega,double gamma,int row,int col,int n)
{
	int i,j,k;
	
	for(i=0;i<row;i++)
	{
		for(j=0;j<col;j++)
		{
			unified[i][j]=0.0;
			for(k=0;k<n;k++)
				unified[i][j]+=(omega[k]*network[k][i][j]);
			unified[i][j]=(final[i][j]+(gamma*unified[i][j]))/(double)(1+gamma);
		}
	}
	
	return;
}

void update_weight(double **unified,double ***network,double **eigenvec,double ***indiv_eigenvec,double *omega,int n,int row,int col,int no_of_clusters)
{
	int i,j,k;
	double term4,term5,term6;
	double sum;
	
	term4=0.0;
	term5=0.0;
	term6=0.0;
	for(k=0;k<n;k++)
	{
		term4=0.0;
		for(i=0;i<row;i++)
			for(j=0;j<row;j++)
				term4+=pow((unified[i][j]-network[k][i][j]),2);
		
		term5=compute_trace2D(network[k],indiv_eigenvec[k],row,no_of_clusters);
		term6=compute_diff_trace(eigenvec,indiv_eigenvec[k],row,no_of_clusters);
		
		sum=term4+term5+term6+0.0001;
		omega[k]=1-sum;
	}
	
	sum=0.0;
	for(k=0;k<n;k++)
		sum+=omega[k];
	
	for(k=0;k<n;k++)
		omega[k]=omega[k]/(double)sum;
	
	return;
}

void compute_individual_eigenvector(double **eigenvec,double **inverse,double **laplacian,double **final_eigenvec,int index,int row,int col,int no_of_clusters,int itr,char *dir)
{
	int i,j,k;
	int flag;
	int negative;
	char fname[1000];
	char comand[1000];
	double **temp;
	struct stat st;
	FILE *fp;
	
	temp=allocate2D(row,col);
	
	printf("Computing inverse using Schur's Complement\n");
	for(i=0;i<row;i++)
	{
		for(j=0;j<row;j++)
			temp[i][j]=laplacian[i][j];
		temp[i][i]=1.0+laplacian[i][i];
	}
		
	write_to_file("temp3.txt",temp,row,row);
	deallocate2D(temp,row);
	
	sprintf(fname,"%s/Inverse%d.txt",dir,index);
	sprintf(comand,"compute_inverse.R");
	compute_inverse("temp3.txt",fname,comand);
	system("R CMD BATCH compute_inverse.R");
	//system("cat compute_inverse.Rout");
	
	fp=fopen(fname,"r");
	for(i=0;i<row;i++)
		for(j=0;j<col;j++)
			fscanf(fp,"%lf",&inverse[i][j]);
	fclose(fp);
	
	flag=checkifsymmetric(inverse,row,col);
	if(flag==0)
		printf("Inverse Not Symmetric\n");
	
	negative=count_negative(inverse,row,col);
	if(negative>0)
	{
		printf("%d negative entries encountered\n",negative);
		printf("WARNING: Inverse not positive definite\n");
		fflush(stdout);
		exit(0);
	}
	
	for(i=0;i<row;i++)
		for(j=0;j<no_of_clusters;j++)
		{
			eigenvec[i][j]=0.0;
			for(k=0;k<col;k++)
				eigenvec[i][j]+=(inverse[i][k]*final_eigenvec[k][j]);
		}
	
	sprintf(fname,"%s/INDIV%d_EIGENVECTOR.txt",dir,index);
	fp=fopen(fname,"w");
	for(i=0;i<row;i++)
	{
		for(j=0;j<no_of_clusters-1;j++)
			fprintf(fp,"%.15lf\t",eigenvec[i][j]);
		fprintf(fp,"%.15lf\n",eigenvec[i][j]);
	}
	fclose(fp);
	
	sprintf(comand,"orthogonalize.R");
	createRscript_GS(fname,fname,comand);
	system("R CMD BATCH orthogonalize.R");
	//system("cat orthogonalize.Rout");
	
	fp=fopen(fname,"r");
	for(i=0;i<row;i++)
		for(j=0;j<no_of_clusters;j++)
			fscanf(fp,"%lf",&eigenvec[i][j]);
	fclose(fp);
	
	return;
}

void compute_individual_eigenvector1(double **eigenvec,double **inverse,double **final_eigenvec,int row,int col,int no_of_clusters,int index,int itr,char *dir)
{
	int i,j,k;
	int flag;
	int negative;
	char fname[1000];
	char comand[1000];
	double del_om_eph;
	double **temp;
	struct stat st;
	FILE *fp;
	
	for(i=0;i<row;i++)
	
		for(j=0;j<no_of_clusters;j++)
		{
			eigenvec[i][j]=0.0;
			for(k=0;k<col;k++)
				eigenvec[i][j]+=(inverse[i][k]*final_eigenvec[k][j]);
		}
	
	sprintf(fname,"%s/INDIV%d_EIGENVECTOR.txt",dir,index);
	fp=fopen(fname,"w");
	for(i=0;i<row;i++)
	{
		for(j=0;j<no_of_clusters-1;j++)
			fprintf(fp,"%.15lf\t",eigenvec[i][j]);
		fprintf(fp,"%.15lf\n",eigenvec[i][j]);
	}
	fclose(fp);
	
	sprintf(comand,"orthogonalize.R");
	createRscript_GS(fname,fname,comand);
	system("R CMD BATCH orthogonalize.R");
	//system("cat orthogonalize.Rout");
	
	fp=fopen(fname,"r");
	for(i=0;i<row;i++)
		for(j=0;j<no_of_clusters;j++)
			fscanf(fp,"%lf",&eigenvec[i][j]);
	fclose(fp);
	
	return;
}

void compute_final_eigenvector(double **eigenvec,double **laplacian,double ***individual_eigenvec,double *omega,double val,int n,int row,int col,int no_of_clusters,int itr,char *dir)
{
	int i,j,k;
	int flag;
	int negative;
	double **temp;
	double **avgeigvec;
	double **inverse;
	char fname[1000];
	char comand[1200];
	FILE *fp;
	
	avgeigvec=allocate2D(row,no_of_clusters);
	for(i=0;i<row;i++)
		for(j=0;j<no_of_clusters;j++)
			for(k=0;k<n;k++)
				avgeigvec[i][j]+=(omega[k]*individual_eigenvec[k][i][j]);
	
	printf("Computing inverse using Schur's Complement\n");
	temp=allocate2D(row,col);
	for(i=0;i<row;i++)
	{
		for(j=0;j<row;j++)
			temp[i][j]=val*laplacian[i][j];
		temp[i][i]=1.0+temp[i][i];
	}
	
	flag=checkifsymmetric(temp,row,col);
	if(flag==0)
		printf("Input Matrix Not Symmetric\n");
	
	write_to_file("temp3.txt",temp,row,row);
	deallocate2D(temp,row);
	
	sprintf(fname,"%s/INVERSE.txt",dir);
	sprintf(comand,"compute_inverse.R");
	compute_inverse("temp3.txt",fname,comand);
	system("R CMD BATCH compute_inverse.R");
	//system("cat compute_inverse.Rout");
	
	inverse=allocate2D(row,row);
	fp=fopen(fname,"r");
	for(i=0;i<row;i++)
		for(j=0;j<row;j++)
			fscanf(fp,"%lf",&inverse[i][j]);
	fclose(fp);
	
	sprintf(comand,"rm %s",fname);
	system(comand);
	
	flag=checkifsymmetric(inverse,row,col);
	if(flag==0)
		printf("Inverse Not Symmetric\n");
	
	negative=count_negative(inverse,row,col);
	if(negative>0)
	{
		printf("WARNING: Inverse not positive definite\n");
		fflush(stdout);
		exit(0);
	}
	
	for(i=0;i<row;i++)
		for(j=0;j<no_of_clusters;j++)
		{
			eigenvec[i][j]=0.0;
			for(k=0;k<col;k++)
				eigenvec[i][j]+=(inverse[i][k]*avgeigvec[k][j]);
		}
	
	sprintf(fname,"%s/FINAL_EIGENVECTOR.txt",dir);
	fp=fopen(fname,"w");
	for(i=0;i<row;i++)
	{
		for(j=0;j<no_of_clusters-1;j++)
			fprintf(fp,"%.15lf\t",eigenvec[i][j]);
		fprintf(fp,"%.15lf\n",eigenvec[i][j]);
	}
	fclose(fp);
	
	sprintf(comand,"orthogonalize.R");
	createRscript_GS(fname,fname,comand);
	system("R CMD BATCH orthogonalize.R");
	//system("cat orthogonalize.Rout");
	
	fp=fopen(fname,"r");
	for(i=0;i<row;i++)
		for(j=0;j<no_of_clusters;j++)
			fscanf(fp,"%lf",&eigenvec[i][j]);
	fclose(fp);
	
	deallocate2D(inverse,row);
	deallocate2D(avgeigvec,row);
	
	return;	
}

void compute_inverse(char *ipfile,char *opfile,char *execR)
{
	FILE *fp;
	
	fp=fopen(execR,"w");
	fprintf(fp,"require(Matrix)\n");
	fprintf(fp,"library(data.table)\n");
	fprintf(fp,"compute_inverse<-function(ipfile,opfile){\n");
	fprintf(fp,"cwd<-getwd()\n");
	fprintf(fp,"fname1<-paste(cwd,ipfile,sep=\"/\")\n");
	fprintf(fp,"fname2<-paste(cwd,opfile,sep=\"/\")\n");
	fprintf(fp,"L<-as.matrix(fread(file=fname1,sep=\"\t\",header=FALSE))\n");
	fprintf(fp,"n=dim(L)[1]\n");
	fprintf(fp,"n1=round(n/3)\n");
	fprintf(fp,"n2=n-n1\n");
	fprintf(fp,"L11=L[1:n1,1:n1]\n");
	fprintf(fp,"L12=L[1:n1,(n1+1):n]\n");
	fprintf(fp,"L21=L[(n1+1):n,1:n1]\n");
	fprintf(fp,"L22=L[(n1+1):n,(n1+1):n]\n");
	fprintf(fp,"L11_Inv=solve(L11)\n");
	fprintf(fp,"L21_L11_Inv=L21 %s L11_Inv\n","%*%");
	fprintf(fp,"L21_L11_Inv_L12 = L21_L11_Inv %s L12\n","%*%");
	fprintf(fp,"F = L22 - L21_L11_Inv_L12\n");
	fprintf(fp,"Identity = diag(1,n2,n2)\n");
	fprintf(fp,"Mat1=matrix(0,n,n2)\n");
	fprintf(fp,"Mat1[1:n1,] = -1 * t(L21_L11_Inv)\n");
	fprintf(fp,"Mat1[(n1+1):n,] = Identity\n");
	fprintf(fp,"Prod1 = Mat1 %s solve(F)\n","%*%");
	fprintf(fp,"Mat2=matrix(0,n2,n)\n");
	fprintf(fp,"Mat2[,1:n1] = -1 * L21_L11_Inv\n");
	fprintf(fp,"Mat2[,(n1+1):n] = Identity\n");
	fprintf(fp,"Prod2 = Prod1 %s Mat2\n","%*%");
	fprintf(fp,"Mat3 =matrix(0,n,n)\n");
	fprintf(fp,"Mat3[1:n1,1:n1] = L11_Inv\n");
	fprintf(fp,"L_Inv = Mat3 + Prod2\n");
	fprintf(fp,"write.table(L_Inv,file=fname2,sep=\"\t\",row.names=FALSE,col.names=FALSE)\n");
	fprintf(fp,"remove(list=ls())\n}\n\n");
	fprintf(fp,"compute_inverse(\"%s\",\"%s\")\n",ipfile,opfile);
	fclose(fp);
	
	return;
}

void createRscript_GS(char *fname,char *fname1,char *execR)
{
	FILE *fp;
	
	fp=fopen(execR,"w");
	fprintf(fp,"library(pracma)\n");
	fprintf(fp,"library(data.table)\n");
	fprintf(fp,"orthogonalize<-function(ipfile,opfile){\n");
	fprintf(fp,"cwd<-getwd()\n");
	fprintf(fp,"fname<-paste(cwd,ipfile,sep=\"/\")\n");
	fprintf(fp,"fname1<-paste(cwd,opfile,sep=\"/\")\n");
	fprintf(fp,"vec<-as.matrix(fread(file=fname,sep=\"\t\",header=FALSE))\n");
	fprintf(fp,"gs = gramSchmidt(vec)\n");
	fprintf(fp,"write.table(gs$Q,file=fname1,sep=\"\t\",row.names=FALSE,col.names=FALSE)\n");
	fprintf(fp,"remove(list=ls())\n}\n\n");
	fprintf(fp,"orthogonalize(\"%s\",\"%s\")\n",fname,fname1);
	fclose(fp);
	
	return;
}

double compute_l1norm(double **mat1,double **mat2,int row,int col)
{
	int i,j;
	double *l1norm;
	double max;
	
	l1norm=(double *)calloc(row,sizeof(double));
	for(i=0;i<col;i++)
		for(j=0;j<row;j++)
			l1norm[i]+=fabs(mat1[i][j]-mat2[i][j]);
	
	max=l1norm[0];
	for(i=0;i<col;i++)
		if(l1norm[i]>max)
			max=l1norm[i];
	
	free(l1norm);
	
	return max;
}
		

void compute_eigenvector(char *ipfile,char *opfile,char *opfile1,int no_of_clusters,char *execR)
{
	FILE *fp;
	
	fp=fopen(execR,"w");
	fprintf(fp,"library(SparseM)\n");
	fprintf(fp,"library(data.table)\n");
	fprintf(fp,"compute_eigenvector<-function(ipfile,opfile,opfile1,no_of_clusters){\n");
	fprintf(fp,"cwd<-getwd()\n");
	fprintf(fp,"fname1<-paste(cwd,ipfile,sep=\"/\")\n");
	fprintf(fp,"fname2<-paste(cwd,opfile,sep=\"/\")\n");
	fprintf(fp,"fname3<-paste(cwd,opfile1,sep=\"/\")\n");
	fprintf(fp,"mat<-as.matrix(fread(file=fname1,sep=\"\t\",header=FALSE))\n");
	fprintf(fp,"eig = eigen(mat)\n");
	fprintf(fp,"res = sort(abs(eig$values),decreasing=FALSE,index.return = TRUE)\n");
	fprintf(fp,"U=eig$vectors[,res$ix[1:no_of_clusters]]\n");
	fprintf(fp,"write.table(U,file=fname2,sep=\"\t\",row.names=FALSE,col.names=FALSE)\n");
	fprintf(fp,"write.table(res,file=fname3,sep=\"\t\",row.names=FALSE,col.names=FALSE)\n");
	fprintf(fp,"remove(list=ls())\n}\n\n");
	fprintf(fp,"compute_eigenvector(\"%s\",\"%s\",\"%s\",%d)\n",ipfile,opfile,opfile1,no_of_clusters);
	fclose(fp);
	
	return;
}

void normalize(double **mat,int row,int col)
{
	int i,j,k;
	double sum;
	
	for(j=0;j<row;j++)
	{
		sum=0.0;
		for(k=0;k<col;k++)
			sum+=mat[j][k];
			
		if(sum!=0.0)
			for(k=0;k<col;k++)
				mat[j][k]=mat[j][k]/(double)sum;
		else
		{
			for(k=0;k<col;k++)
				mat[j][k]=0.0;
			/*printf("Error Encountered...All zero-row found\n");
			printf("j=%d\tk=%d\n",j,k);
			fflush(stdout);
			exit(0);*/
		}
	}
	
	return;
}

void symmetricize(double **mat,int row,int col)
{
	int i,j,k;
	
	for(j=0;j<row;j++)
	{
		for(k=j;k<col;k++)
		{
			mat[j][k]=(mat[j][k]+mat[k][j])/(double)2;
			mat[k][j]=mat[j][k];
		}
	}
	
	return;
}
	
double ***allocate3D(int n,int row,int col)
{
	int i,j,k;
	double ***mat;
	
	mat=(double ***)malloc(n*sizeof(double **));
	if(mat!=NULL)
	{
		for(i=0;i<n;i++)
		{
			mat[i]=(double **)malloc(row*sizeof(double *));
			if(mat[i]!=NULL)
			{
				for(j=0;j<row;j++)
				{
					mat[i][j]=(double *)calloc(col,sizeof(double));
					if(mat[i][j]==NULL)
					{
						printf("Insufficient Memory\n");
						exit(0);
					}
				}
			}
		}
	}
	//printf("Created a matrix of dimension %d x %d x %d\n",n,row,col);
	
	return mat;
}

double **allocate2D(int row,int col)
{
	int i;
	double **mat;
	
	mat=(double **)malloc(row*sizeof(double *));
	for(i=0;i<row;i++)
		mat[i]=(double *)calloc(col,sizeof(double));
	//printf("Created a matrix of dimension %d x %d\n",row,col);
		
	return mat;
}

void deallocate3D(double ***mat,int n,int row)
{
	int i,j;
	
	for(i=0;i<n;i++)
	{
		for(j=0;j<row;j++)
			free(mat[i][j]);
		free(mat[i]);
	}
	free(mat);
	
	return;
}

void deallocate2D(double **mat, int row)
{
	int i;
	
	for(i=0;i<row;i++)
		free(mat[i]);
	free(mat);
	
	return;
}	
