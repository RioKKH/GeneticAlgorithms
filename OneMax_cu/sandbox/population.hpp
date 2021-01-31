class population
{
public:
    population();
    ~population();

private:
    int *h_ind, *h_next_ind, *h_fitness, *h_temp_ind;
    int *d_ind, *d_next_ind, *d_fitness, *d_temp_ind;

    int pop_num;
    int chromosome_num;
};



