#include "fof.hpp"

enum FOF_STATE : size_t {
    NO_GROUP        = size_t(-1),
    NOT_PROCESSED   = size_t(-2),
    PROCESSING      = size_t(-3),
};

extern "C"
void find_fof_groups(size_t N,
                     double *pos,
                     double *vel,
                     double *mass,
                     double l,
                     double dvmax,
                     size_t min_parts,
                     int sort,
                     size_t *FoF,
                     double periodic,
                     void *octree) {
    //printf("perform FoF finder (ll=%.3g, N>=%zu)...\n", l, min_parts);
    assert(N < PROCESSING);
    double dvmax2 = std::pow(dvmax,2.0);

    Tree<3> *tree = NULL;
    if (octree == NULL) {
        //printf("initizalize tree...\n");
        tree = (Tree<3> *)new_octree_from_pos(N, pos);
        assert(tree);
    } else {
        tree = (Tree<3> *)octree;
    }

    //printf("preflag all particles as not yet processed...\n");
    for (size_t i=0; i<N; i++) {
        FoF[i] = NOT_PROCESSED;
    }

    //printf("do actual FoF finding...\n");
    size_t group_idx = 0;
    std::vector<double> FoF_mass;
    for (size_t i=0; i<N; i++) {
        // already in some group
        if (FoF[i] != NOT_PROCESSED) {
            assert(FoF[i] != PROCESSING);
            continue;
        }

        size_t Npart = 0;
        double M = 0.0;
        // find all friends
        std::vector<size_t> del_FoF;
        std::vector<size_t> friends(1,i);
        assert(friends[0]==i);
        FoF[i] = PROCESSING;
        while (friends.size()) {
            size_t j = friends.back();
            friends.pop_back();
            if (Npart < min_parts)
                del_FoF.push_back(j);
            // do not want to process particles twice!
            assert(FoF[j] == PROCESSING);

            FoF[j] = group_idx;
            Npart++;
            M += mass[j];

            std::vector<size_t> new_friends
                = tree->ngbs_within_if(pos+3*j, l, pos, periodic,
                                       [&FoF,&vel,&j,&dvmax2](size_t idx){
                    return FoF[idx]==NOT_PROCESSED && dist2<3>(vel+3*j, vel+3*idx)<dvmax2;
            });
            // avoid finding particles twice -> pretag with PROCESSING
            for (const size_t k : new_friends) {
                assert(FoF[k] == NOT_PROCESSED);    // check condition of search
                FoF[k] = PROCESSING;
            }
            // append the new friends that are not yet processed
            friends.insert(friends.end(),
                           new_friends.begin(),new_friends.end());
        }
        // done with this group
        if (Npart < min_parts) {
            for (const size_t j : del_FoF) {
                assert(FoF[j] == group_idx);
                FoF[j] = NO_GROUP;
            }
        } else {
            //printf("Done /w #%zu -- %zu particles & mass=%g\n", group_idx, Npart, M);
            group_idx++;
            FoF_mass.push_back(M);
        }
    }
    //printf("Found %zu groups with at least %zu particles.\n", group_idx, min_parts);

    // sort halos by mass
    if (sort) {
        //printf("sort halos by mass...\n");
        std::vector<size_t> group(FoF_mass.size());
        for (size_t i=0; i<FoF_mass.size(); i++) {
            group[i] = i;
        }
        // sort group indices descending by mass
        std::sort(group.begin(), group.end(),
                  [&FoF_mass](size_t i, size_t j){return FoF_mass[i] > FoF_mass[j];});

        /*
        printf("The %zu biggest FoF halos:\n", std::min<size_t>(FoF_mass.size(), 10));
        for (size_t i=0; i<std::min<size_t>(FoF_mass.size(), 10); i++) {
            printf("  halo %5zu: %8.3g\n", group[i], FoF_mass[group[i]]);
        }
        */

        // invert the mapping of `group` (now is i-th biggest -> group ID)
        std::vector<size_t> new_ID(group.size());
        for (size_t i=0; i<group.size(); i++) {
            new_ID[group[i]] = i;
        }

        //printf("update FoF group indices for particles...\n");
        for (size_t i=0; i<N; i++) {
            if (FoF[i] == NO_GROUP)
                continue;
            assert(FoF[i] != PROCESSING and FoF[i] != NOT_PROCESSED);
            FoF[i] = new_ID[FoF[i]];
        }
    }

    if (octree == NULL) {
        //printf("delete tree...\n");
        delete tree;
    }
}
