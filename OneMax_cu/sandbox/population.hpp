class population
{
public:
    population();
    ~population();

private:
    int *h_ind, *h_next_ind, *h_fitness, *h_temp_ind, *h_diag;
    int *d_ind, *d_next_ind, *d_fitness, *d_temp_ind, *d_diag;

    int pop_num;
    int chromosome_num;
};



