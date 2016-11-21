#include <cstdio>
#include <algorithm>
#include <ctime>
#include <cstdlib> 
#include <thread> 
#define CORE_NUM 4
using namespace std;  

int T,K,M,N;

inline double getR(int x1, int y1, int x2, int y2, double ** gs, double ** model){
	 return gs[x2][y2] > 0
	 	? (gs[x1][y1]/model[x1][y1])/(gs[x2][y2]/model[x2][y2])
	 	: 1e20;
}

int *generate(int n, int max, int *g) {
    int i, m, a;

    m = 0;
    for (i=0; i<max &&  m<n; i++) {
        a = rand()%(max-i);
        if (a < n - m) {
            g[m] = i;
            m ++;
        }
    }
    return g;
 }

inline double _label(int x, int y, double **gs, double **model, int *can) {
	int numLits = 0;
	int numShadows = 0;
	generate(K,M*N,can);
	for (int i=0; i<K; ++i) {
		int pos = can[i];
		double R = getR(x,y,pos/M,pos%M,gs,model);
		if (R>T) numLits++;
		else if (R<1.0/T) numShadows++;
	}
	return double(numLits)/(numShadows+numLits);
}

void _label_thread(int id, double **lbs, double ** gs, double ** model) {
	srand(time(0)*(id+1));
	int *can = new int[K];
	for (int i=id; i<M; i+=CORE_NUM){
		if (id == 0)printf("."),fflush(stdout);
		for (int j=0; j<N; ++j) {
			lbs[i][j] = _label(i,j,gs,model,can);
		}	
	}
	delete [] can;
}

extern "C" {
	void label(const int m, const int n,  double ** model, double ** gs, double ** lbs, int _T, int _K) {
		T = _T; K = _K; M = m; N = n;
		thread t[CORE_NUM];
		printf(">>>>> Calc Labels ...\n");
		for (int i=0; i<CORE_NUM; ++i){
			t[i] = thread(_label_thread,i,lbs,gs,model);
		}
		for (int i=0; i<CORE_NUM; ++i) t[i].join();
		printf("\n");
	}
}
