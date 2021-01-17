// 標準ヘッダーのインクルード
#include <time.h>
#include <stdlib.h>
#include <limits>
#include <float.h>
#include <math.h>


// 定数の定義
#define GEN_MAX         1000    // 世代交代数
#define POP_SIZE        1000    // 個体群のサイズ
#define ELITE           1       // エリート保存戦略で残す個体の数
#define MUTATE_PROB     0.01    // 突然変異確率
#define N               100     // Number of One Max Length
#define TOURNAMENT_SIZE 5       // Tournament size

// 0以上1以下の実数乱数
#define RAND_01 ((double) rand() / RAND_MAX)

class individual
{
public:
    individual();
    ~individual();
    void evaluate();
    // Single-point crossover
    void apply_crossover_sp(individual *p1, individual *p2);
    // Two-point crossover
    void apply_crossover_tp(individual *p1, individual *p2);
    // Uniform corssover
    void apply_crossover_uniform(individual *p1, individual *p2);
    // mutation
    void mutate();
    // chromosome
    int chromosome[N];
    // fitness
    int fitness;
};