#include <cstdio>
#include <algorithm>
#include <ctime>
#include <cstdlib> 
#include <thread> 
#include <unordered_set>
#define CORE_NUM 4
using namespace std;  

int T,K,M,N;

inline double getR(int x1, int y1, int x2, int y2, double ** gs, double ** model){
	 return gs[x2][y2] > 0
	 	? (gs[x1][y1]/model[x1][y1])/(gs[x2][y2]/model[x2][y2])
	 	: 1e20;
}

/*
Gate: http://stackoverflow.com/questions/2394246/algorithm-to-select-a-single-random-combination-of-values/2394308#2394308
initialize set S to empty
for J := N-M + 1 to N do
    T := RandInt(1, J)
    if T is not in S then
        insert T in S
    else
        insert J in S
*/
double generate_label_fast(const int n, const int max, unordered_set<int> &g, const int x, const int y, double **gs, double **model) {
    int m=0;
    int numLits = 0;
	int numShadows = 0;
	double R;
	g.clear();

	for (int i=max-n; i<max; ++i){
		int t = rand()%i;
		if (g.find(t) == g.end()) {
			g.insert(t);
			R = getR(x,y,t/M,t%M,gs,model);
			if (R>T) numLits++;
			else if (R<1.0/T) numShadows++;
		} else {
			g.insert(i);
			R = getR(x,y,i/M,i%M,gs,model);
			if (R>T) numLits++;
			else if (R<1.0/T) numShadows++;
		}
	}
	return double(numLits)/(numShadows+numLits);
 }

// double generate_label(int n, int max, int *g, int x, int y, double **gs, double **model) {
//     int m=0,a;
//     int numLits = 0;
// 	int numShadows = 0;
// 	double R;
//     for (int i=0; i<max &&  m<n; i++) {
//         a = rand()%(max-i);
//         if (a < n - m) {
//             g[m] = i;
//             R = getR(x,y,i/M,i%M,gs,model);
// 			if (R>T) numLits++;
// 			else if (R<1.0/T) numShadows++;
//             m++;
//         }
//     }
//     return double(numLits)/(numShadows+numLits);
//  }

void _label_thread(int id, double **lbs, double ** gs, double ** model) {
	srand(time(0)*(id+1));
	unordered_set<int> can;
	// int *can = new int[K];
	for (int i=id; i<M; i+=CORE_NUM){
		if (id == 0)printf(" Labeling %d%%\r",(i+4)*100/M),fflush(stdout);
		for (int j=0; j<N; ++j) {
			lbs[i][j] = generate_label_fast(K,M*N,can,i,j,gs,model);
			// lbs[i][j] = generate_label(K,M*N,can,i,j,gs,model);
		}	
	}
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
