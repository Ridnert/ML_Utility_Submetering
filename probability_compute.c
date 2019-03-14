#include <stdio.h>
#include <math.h>

/* 
This Program Evaluates the probabilities that the number of outcomes for one class in a multinomial distribution is larger than the others for some 
small value of the number of classes. 
*/


double factorial(int n){
  int c;
  double fact = 1;
  for (c = 1; c <= n; c++)
    fact = fact * c;
  return fact;
}


int main(){
    int twoclasses = 1;
    int threeclasses = 1;
    int fourclasses = 1;

    int maxK = 50;
    int K;
    int n1;
    int n2;
    int n3;
    int n4;
    double p = 0.3;
    double epsilon;
    double eps1;
    double eps2;
    int num_swipes = 5;
    FILE * fp;
    FILE * fp3;
    FILE * fp4class;
    remove("/home/carl/Documents/ML_Utility_Submetering/probabilities.txt");
    remove("/home/carl/Documents/ML_Utility_Submetering/probabilities3.txt"); 
    fp = fopen("/home/carl/Documents/ML_Utility_Submetering/probabilities.txt","w"); 
    fp3 = fopen("/home/carl/Documents/ML_Utility_Submetering/probabilities3.txt","w"); 
   
    /* The worst case scenario where we have two classes */
    for(K=1;K<=maxK;K++){
        double sum = 0;
        for(n1= ceil(K/2) ; n1 <= K; n1++) {
            for(n2 = 0; n2 < n1; n2++){
                    if(n1+n2 == K){
                        sum += (double) factorial(K)*(pow(p,(double)n1) *  pow(1-p,(double)n2) ) / (double)((factorial(n1)*factorial(n2))) ; 
                    }
                }
        }
        fprintf(fp,"%f\n",sum);
    }
    fclose(fp);

    if (threeclasses == 1 ){
    epsilon = 0;
        for(int eps=0;eps < num_swipes; eps++){  
            /* Better scenario, three classes */    
            for(K=1;K <= maxK ;K++){
                double sum = 0;
                for(n1= 0 ; n1 <= K; n1++) {
                    for(n2 = 0; n2 < n1; n2++){
                        for(n3 = 0; n3 < n1; n3++){
                            if(n1+n2+n3 == K){
                                sum += (double) factorial(K)*(pow(p,(double)n1) *  pow((1-p-epsilon),(double)n2) * pow(epsilon,(double)n3)) / (double)((factorial(n1)*factorial(n2) *factorial(n3) ) ) ; 
                            }
                        }
                    }
                }
                fprintf(fp3,"%f\n",sum);  
            }
            epsilon = epsilon + 0.15/num_swipes;
        }
    }
    
    fclose(fp3);
    /* Adding Fourth Class */
    if(fourclasses == 1){
        remove("/home/carl/Documents/ML_Utility_Submetering/probabilities4class.txt");
        fp4class = fopen("/home/carl/Documents/ML_Utility_Submetering/probabilities4class.txt","w");
        eps1 = 0;
        for (int e1 = 0; e1 < 10;e1++ ){
            eps1 = eps1 + 0.15/10;
            eps2 = 0;
            for(int e = 0; e < num_swipes; e++){
            /* Four Classes */    
                for(K=1;K <= maxK ;K++){
                    double sum = 0;
                    for(n1= ceil(K/4) + 1 ; n1 <= K; n1++) {
                        for(n2 = 0; n2 < n1;n2++){
                            for(n3 = 0; n3 < n1;n3++){
                                for(n4 = 0; n4 < n1; n4++){
                                    if(n1+n2+n3+n4 == K){
                                        sum += (double) factorial(K)*(pow(p,(double)n1) *  pow(1-p-(eps1+eps2),(double)n2) * pow(eps1,(double)n3) * pow(eps2,(double)n4)) / 
                                            (double)((factorial(n1)*factorial(n2) *factorial(n3) *factorial(n4)) ) ; 
                                    
                                    }
                                }
                            }
                        }
                    }
                    fprintf(fp4class,"%f\n",sum);
                }
                eps2 = eps2 + (0.15-eps1)/num_swipes;
            }
        }
        fclose(fp4class);
    }
    return 0;
}