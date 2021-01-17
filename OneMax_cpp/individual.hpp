// 標準ヘッダーのインクルード
#include <time.h>
#include <stdlib.h>
#include <limits>
#include <float.h>
#include <math.h>

#include "parameters.hpp"


// 定数の定義
/*
#define GEN_MAX         10      // 世代交代数
#define POP_SIZE        100     // 個体群のサイズ
#define ELITE           1       // エリート保存戦略で残す個体の数
#define MUTATE_PROB     0.01    // 突然変異確率
#define N               10      // Number of One Max Length
#define TOURNAMENT_SIZE 5       // Tournament size
*/

// 0以上1以下の実数乱数
#define RAND_01 ((double) rand() / RAND_MAX)

class individual
{
public:
    // individual();
    individual(Parameters *prms);
    ~individual();
    void evaluate();
    void load_params();
    // Single-point crossover
    void apply_crossover_sp(individual *p1, individual *p2);
    // Two-point crossover
    void apply_crossover_tp(individual *p1, individual *p2);
    // Uniform corssover
    void apply_crossover_uniform(individual *p1, individual *p2);
    // mutation
    void mutate();
    // chromosome
    int *chromosome;
    // int chromosome[N];
    // fitness
    int fitness;
};
