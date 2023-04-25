#include "mixnet.h"
#include <vector>
#include <cstring>
#include <iostream>
#include <random>
#include <cstdalign>
#include <algorithm>
#include <string>
#include <map>

const int inf_w = 1<<25;

unsigned seed = 1234567;
std::default_random_engine generator(seed);
std::normal_distribution<float> distribution(0.0, 1.0);

void szero(float *v, int l)
{
    memset(v, 0, l*sizeof(*v));
}

float sdot (const float* x, const float* y, int l) {
    float res = 0.0;
    for (int i = 0; i < l; ++i) {
        res += x[i] * y[i];
    }
    return res;
}

struct Lit {
    std::string var;
    bool positive;
    Lit(bool p, std::string s) : positive (p), var(s) {}

    void show() {
        if (positive) {
            printf("     %s ", var.c_str());
        }
        else {
            printf("(not %s)", var.c_str());
        }
    }

    Lit negate() {
        Lit tmp = *this;
        tmp.positive = ! tmp.positive;
        return tmp;
    }
};

struct Clause {
    std::vector<Lit> cl;

    Clause () {}
    Clause(std::initializer_list<Lit> init): cl(init){}

    void addLit(Lit && l) {
        cl.push_back(l);
    }

    void show(){
        for (auto& l : cl) {
            l.show();
            printf(" ");
        }
    }

};

struct SATInstance {
    std::vector<Clause> cls;
    std::vector<int> weights;

    void addClause(Clause&& cl, int w=1) {
        cls.push_back(cl);
        weights.push_back(w);
    }

    void show() {
        printf("SAT Instance with %ld clauses:\n", cls.size());
        for(int i = 0; i < cls.size(); ++i){
            cls[i].show();
            if (weights.at(i) == inf_w) {
                printf(" (hard clause)\n");
            }
            else{
                printf(" (soft w=%d)\n", weights.at(i));
            }
        }
    }

    void to_sat_encoding (FILE* f) {
        std::map<std::string, int> mp;
        std::vector<std::vector<int>> encodings;

        for (int i =0; i < cls.size(); ++i) {
            std::vector<int> v;
            v.push_back(weights.at(i));

            for(auto& l : cls[i].cl) {
                if(mp.find(l.var) == mp.end()) {
                    int index = mp.size() + 1;
                    mp[l.var] = index;
                }

                v.push_back(l.positive ? mp[l.var] : -mp[l.var]);
            }
            encodings.push_back(std::move(v));
        }

        // http://maxsat.ia.udl.cat/requirements/
        // "p wcnf nbvar nbclauses".  (weighted)
        fprintf(f, "p wcnf %ld %ld\n", mp.size(), encodings.size());
        for(auto it = mp.begin(); it != mp.end(); ++it) {
            fprintf(f, "c %s ==> %d\n", it->first.c_str(), it->second);
        }
        for(auto& v : encodings) {
            for(auto& x : v) fprintf(f, "%d ",x);
            fprintf(f, "0\n");
        }
    }

};

struct AdamSNet {
    const float eps = 1e-8;
    float alpha = 0.01;   // step size
    float *m, beta1 = 0.9, beta1_t=1.0; // 1st order moment and decay
    float *v, beta2 = 0.999, beta2_t=1.0; // 2nd order moment and decay
    const int rows, cols;

    AdamSNet(int r, int c) : rows(r), cols(c) {
        m = new float [r * c];
        v = new float [r * c];
        szero(m, r * c);
        szero(v, r * c);
    }

    void step(float *theta, const float *grad) {
        beta1_t *= beta1;
        beta2_t *= beta2;

        // skip any update for the truth vector (i.e., 0-th row)
        for (int r = 0; r < rows; ++r) {
            for(int c = 0; c < cols; ++c) {
                const int index = r * cols + c;
                const float g = grad[index];
                const float g2 = g * g;

                m[index] = beta1 * m[index] + (1.0 - beta1) * g;
                v[index] = beta2 * v[index] + (1.0 - beta2) * g2;

                const float c_m = m[index] / (1.0 - beta1_t);
                const float c_v = v[index] / (1.0 - beta2_t);

                const float delta = alpha * c_m / (eps + sqrt(c_v));
                if(delta > 0.01) {
                    printf("delta=%f is too large!\n", delta);
                }

                theta[index] -= delta;
            }
        }

    }

};

struct Learner
{
    int nvar, aux;

    std::vector<std::vector<bool>> assignments;
    std::vector<int> input_dims;
    std::vector<bool> io_dims;
    std::vector<int> perm;

    Learner(std::vector<int>&& ins, std::vector<std::vector<bool>>&& assg, int a = 0) : input_dims(ins), assignments(assg), aux(a) {
        const int n = assignments[0].size() + a + 1; // one extra variable for the truth vector
        nvar = n;
        perm.clear();
        io_dims.clear();
        for (int i = 0; i < n; ++i) {
            if(i < n-1){
                perm.push_back(i);
            }
            io_dims.push_back(false);
        }
        io_dims[0] = true;
        for (auto x : input_dims) {
            io_dims[x] = true;
        }
    }

    void construct_mix_params(mix_t &params)
    {
        const int n = nvar;
        const int k = 16; // embedding dimension size
        const int b = assignments.size(); // batch
        params.n = n;
        params.b = b;
        params.k = k;

        // input/output learning signals
        params.is_input = new int32_t[b * n];
        memset(params.is_input, 0, b * n * sizeof(int32_t));

        params.z = new float[b * n];

        for (int j = 0; j < b; ++j) {
            params.z[j * n] = 1.0;
            for (int i = 1; i < n; ++i){
                if (i <= assignments[j].size()) {
                    params.z[j * n + i] = assignments[j].at(i-1) ? 1.0 : 0.0;
                }
                else {
                    // assignment for auxiliary variables
                    params.z[j * n + i] = 0.5;
                }
            }
            params.is_input[j * n] = 1;
            for (auto x : input_dims) {
                params.is_input[j * n + x] = 1;
            }
        }
        params.dz = new float[b * n];

        // mixing method update schedule
        params.index = new int32_t[b * n];

        params.niter = new int32_t[b];
        for (int i = 0; i < b; ++i) {
            params.niter[i] = 40;
        }

        const float r = 0.1;
        params.C = new float[n * n];
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < i; ++j) {
                int index1 = i * n + j;
                int index2 = j * n + i;
                float t = r * distribution(generator);
                params.C[index1] = params.C[index2] = t;
            }
        }

        /*
        std::vector<std::vector<float>> gt = {
        {  0,   -2.58, -3.47, 1.61, -0.21, 6.84, },
        {-2.58,   0,   2.82, -1.18, 0.29, -5.45, },
        {-3.47, 2.82,   0,   -1.58, 0.70, -6.99, },
        {1.61, -1.18, -1.58,   0,   -0.48, 2.62, },
        {-0.21, 0.29, 0.70, -0.48,   0,   0.11, },
        {6.84, -5.45, -6.99, 2.62, 0.11,   0,   },
        };

        for(int i=0; i < n; ++i){
            for(int j =0; j < n; ++j){
                const int index = i * n + j;
                params.C[index] = gt[i].at(j);
            }
        }*/


        params.dC = new float[ b * n * n];

        // params.Snrms = new float[n];
        // compute_Snrms(params);


        // W and Phi are NOT needed in MixNet
        // params.W = new float[b * k * m];
        // params.Phi = new float[b * k * m];

        params.V = new float [b * n * k];
        for (int i = 0; i < b * n * k; ++i) {
            params.V[i] = distribution(generator);
        }
        params.U = new float [b * n * k];

        params.cache = new float[b * k];
        params.gnrm = new float[b * n];
    }

    void compute_dz(mix_t& params ) {
        const int n = params.n;
        const int b = params.b;
        szero(params.dz, b * n);

        for (int j = 0; j < b; ++j){
            for (int i = 0; i < n; ++i) {
                if(io_dims[i]) continue;
                const int index = j * n + i;
                if (i <= assignments[j].size()){
                    params.dz[index] = 2 * params.z[index] - 2.0 * (assignments[j].at(i-1) ? 1.0 : 0.0);
                }

                //printf("batch = %d, dloss/dv[%d] = %.6f ", j, i, params.dz[index]);
            }
            //printf("\n");
        }
    }

    float compute_loss(mix_t& params ) {

        float loss = 0.0;
        const int n = params.n;
        for (int j = 0; j < params.b; ++j){
            float b_loss = 0.0;
            for (int i = 0; i < n; ++i) {
                if(io_dims[i]) continue;
                const int index = j * n + i;
                if (i <= assignments[j].size()){
                    float groundtruth = (assignments[j].at(i-1) ? 1.0 : 0.0);
                    float delta = (params.z[index] - groundtruth);
                    b_loss =  delta * delta;
                }
            }
            //printf("batch=%d, b_loss=%f\n", j, b_loss);
            loss += b_loss;
        }
        printf("total loss: %f\n", loss);
        return loss;
    }

    void show_dC(mix_t& params) {
        const int n = params.n;
        const int b = params.b;
        for (int _b = 0 ; _b < b; ++ _b) {
            printf("batch=%d, dC:\n", _b);
            dbgout2D("", params.dC + _b * n * n, n, n, "\n");
        }
    }

    void accumulate_dC(mix_t& params) {
        const int n = params.n;
        const int b = params.b;
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                float r = 0.0;
                const int index = i * n + j;
                for (int _b = 0 ; _b < b; ++ _b) {
                    r += params.dC[_b * n * n + index];
                }
                params.dC[index] = r;
            }
        }
    }

    void forward(mix_t& params){
        std::random_shuffle(perm.begin(), perm.end());
        //perm = { 1, 2, 3, 0};
        mix_init_launcher_cpu(params, perm.data());

        //prepare_W(params);

        printf("initial values, V:\n");
        for (int i=0; i< params.b; ++i) {
            printf("for batch = %d: ", i);
            const int base = i * params.n * params.k;
            for (int j = 0; j < params.n; ++j){
                printf("%.2f ", sdot(params.V+base, params.V+base + j * params.k, params.k));
            }
            printf("\n");
        }

        // dbgout2D("in forward, before mixing, S:\n", params.S, params.n, params.m);

        mix_forward_launcher_cpu(params, 40, 1e-10);

        // printf("after mixing, V:");
        // for (int i=0; i< params.b; ++i) {
        //     printf("for batch = %d\n", i);
        //     const int base = i * params.n * params.k;
        //     dbgout2D("", params.V + base, params.n, params.k);
        // }

        dbgout2D("\npredicted outputs after forward process:\n", params.z, params.b, params.n, "\n");
        // dbgout2D("gnrm:\n", params.gnrm, params.b, params.n);

    }
    void grad_zero(mix_t& params) {
        const int n = params.n;
        const int b = params.b;
        const int k = params.k;

        szero(params.dC, b * n * n);
    }
    void backward(mix_t& params) {

        // dz depends on the downstream loss function
        compute_loss(params);
        printf("\n");
        compute_dz(params);


        float prox_lam = 0.01;

        //show_dS(params);
        // compute gradient using Mixing method
        mix_backward_launcher_cpu(params, prox_lam);



        // update S
        // update_S(params);
        // dbgout2D("updated S:\n", params.S, params.n, params.m);
        // update Snrms
        // compute_Snrms(params);

    }

    void learn() {
        mix_t params;
        construct_mix_params(params);
        // AdamSNet adam(params.n, params.m);
        AdamSNet adam(params.n, params.n);

        for(int iter = 0; iter < 100; ++iter) {
            forward(params);

            float loss =  compute_loss(params);
            if(loss < 1e-5) {
                printf("loss %f is sufficiently small, iter = %d\n", loss, iter);
                break;
            }

            grad_zero(params);

            backward(params);

            //dbgout2D("show_dS_sum:\n", params.dS_sum, params.n, params.m);

            // compute_dS_sum(params);

            // dbgout2D("show_dS_sum:\n", params.dS_sum, params.n, params.m);

            // accumulate all gradients to the first batch  b * n * n -> n * n
            accumulate_dC(params);
            dbgout2D("show_dC_sum:\n", params.dC, params.n, params.n, "\n");

            dbgout2D("before step, C:\n", params.C, params.n, params.n, "\n");

            //adam.step(params.S, params.dS_sum);
            adam.step(params.C, params.dC);

            dbgout2D("after step, C:\n", params.C, params.n, params.n, "\n");

            //update_S(params);

            // update Snrms
            // compute_Snrms(params);

            //dbgout2D("after one round forward and backward, S:\n", params.S, params.n, params.m);
            int dbg = 0;
        }
    }

    void solve() {
        mix_t params;
        construct_mix_params(params);
        forward(params);

        float loss =  compute_loss(params);

        dbgout2D("C:\n", params.C, params.n, params.n, "\n");

        // gen_sat(params);
        SATInstance s = interpret_by_C(params);
        FILE* fout = fopen("aw.txt", "w");
        if (fout != nullptr) {
            s.to_sat_encoding(fout);
        }
    }

    bool close_to_zero(float v) {
        const float e = 0.1;
        return v > -e && v < e;
    }

    SATInstance interpret_by_C(mix_t& params){
        // C = S * S'
        // C is symmetric
        const int n = params.n;

        SATInstance s;

        std::vector<std::string>  names;
        for (int i = 0; i < n; ++i) {
            std::string t = i < n - aux ? "v" + std::to_string(i) : "a" + std::to_string(i - n + aux + 1);
            names.push_back(t);
        }

        // truth variable has to be true
        s.addClause({Lit(true, names[0])}, inf_w);

        for (int i = 0; i < n; ++i) {
            // C is symmetric, only need to compute half
            for (int j = 0; j < i; ++j) {
                // if (i == j) continue;
                if (io_dims[i] && io_dims[j]) continue;

                // int w = int (100 * sdot(params.S+i*m, params.S + j * m, m));
                int w = int (100 * params.C[i * n + j]);

                std::string res = "Eq_" + names[i] + "_" + names[j] + "_w=" + std::to_string(w);
                s.addClause({Lit(false, names[i]), Lit(false, names[j]), Lit(true, res)}, inf_w);
                s.addClause({Lit(true, names[i]), Lit(true, names[j]), Lit(true, res)}, inf_w);
                s.addClause({Lit(false, names[i]), Lit(true, names[j]), Lit(false, res)}, inf_w);
                s.addClause({Lit(true, names[i]), Lit(false, names[j]), Lit(false, res)}, inf_w);

                // s.addClause({Lit(true, res)}, max_weight - w);
                // s.addClause({Lit(false, res)}, max_weight + w);

                // compute minimal weighted sum
                if (w > 0) {
                    s.addClause({Lit(false, res)},  w + w);
                }
                else {
                    s.addClause({Lit(true, res)},  -w - w);
                }

            }
        }

        return std::move(s);
    }

};

int main()
{
    // x XOR y =z
    std::vector<int> input_dims = {1, 2};
    std::vector<std::vector<bool>> assignments = {
        {false, false, false},
        {true, true, false},
        {true, false, true},
        {false, true, true},
        }; // a bit strange, the order matters!!

    // x /\ y => z
    // one rule:  not x \/ not y \/ z
    // std::vector<int> input_dims = {1, 2};
    // std::vector<std::vector<bool>> assignments = {
    //     {true, true, true},
    //     {false, true, true},
    //     {true, false, true},
    //     {false, false, true},
    //     {false, false, false},
    //     {true, false, false},
    //     {false, true, false},
    // };

    // x /\ y <==> z
    // not x \/ not y \/ z
    // not z \/ x
    // not z \/ y
    // std::vector<int> input_dims = {1, 2};
    // std::vector<std::vector<bool>> assignments = {
    //     {false, true, false},
    //     {true, false, false},
    //     {true, true, true},
    //     {false, false, false},
    // };


    Learner L(std::move(input_dims), std::move(assignments), 2);

    L.learn();
    // L.solve();

    return 0;
}