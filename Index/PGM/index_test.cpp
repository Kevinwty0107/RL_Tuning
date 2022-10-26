#include <vector>
#include <cstdlib>
#include <iostream>
#include <algorithm>
#include <time.h>
#include <stdio.h>
#include <cstring>
#include <stdlib.h>
#include <fstream>
#include <cstdlib>
#include <sstream>


using namespace std;


#include "pgm/pgm_index.hpp"
#include "pgm/pgm_index_dynamic.hpp"
#include "pgm/pgm_index_variants.hpp"

template<typename K>
class PGMMultiset {
    std::vector<K> data;
    pgm::PGMIndex<K,64, 4, float> pgm;

public:

    explicit PGMMultiset(const std::vector<K> &data) : data(data), pgm(data.begin(), data.end()) {}

    bool contains(const K x) const {
        auto range = pgm.search(x);
        return std::binary_search(data.begin() + range.lo, data.begin() + range.hi, x);
    }

    auto lower_bound(const K x) const {
        auto range = pgm.search(x);
        return std::lower_bound(data.begin() + range.lo, data.begin() + range.hi, x);
    }

    auto upper_bound(const K x) const {
        auto range = pgm.search(x);
        auto it = std::upper_bound(data.begin() + range.lo, data.begin() + range.hi, x);
        auto step = 1ull;
        while (it + step < end() && *(it + step) == x)  // exponential search to skip duplicates
            step *= 2;
        return std::upper_bound(it + (step / 2), std::min(it + step, end()), x);
    }

    size_t count(const K x) const {
        auto lb = lower_bound(x);
        if (lb == end() || *lb != x)
            return 0;
        return std::distance(lb, upper_bound(x));
    }

    auto begin() const { return data.cbegin(); }
    auto end() const { return data.cend(); }
};


template<typename T, typename Iter>
void print_iterator(Iter first, Iter last) {
    std::cout << "[";
    if (first != last) {
        std::copy(first, std::prev(last), std::ostream_iterator<T>(std::cout, " "));
        std::cout << *std::prev(last);
    }
 
 
    std::cout << "]" << std::endl;
};

// 
// vector<string> readTXT();




int main(int argc, char * argv[]) {

        string labels_txt_file = argv[1];
        std::vector<string> data;

	    std::ifstream fp(labels_txt_file);
	    if (!fp.is_open())
	    {
		    printf("could not open file...\n");
		    exit(-1);
	    }
	    std::string name;
	    while (!fp.eof())
	    {
		    std::getline(fp, name);
		    if (name.length())
			    data.push_back(name);
	    }
	    fp.close();
    
        std::vector<float> new_data;

        for(auto it = data.begin(); it != data.end(); it++)
        {
            new_data.push_back(stof(*it));
        }

        std::sort(new_data.begin(), new_data.end());

    // std::cout << "data = ";
    
    // print_iterator<int>(new_data.begin(), new_data.end());


        PGMMultiset multiset(new_data);

        clock_t start,end;
    // Query the PGM-index

    // sample queries
        start=clock();
        for (int i=0;i<2000;i++){

            multiset.count(i);
            (multiset.contains(i) ? "true" : "false");
            *multiset.lower_bound(i);
            *multiset.upper_bound(i);

        }

        end=clock();

        ofstream myout("./runtime_result.txt");
        myout<< (double)(end-start)/CLOCKS_PER_SEC << endl;
        myout.close();
    // std::cout << "Range search [50, 60] = ";
    // print_iterator<int>(multiset.lower_bound(50), multiset.upper_bound(60));
        return 0;

}

