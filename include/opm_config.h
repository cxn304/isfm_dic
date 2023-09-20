#pragma once
#ifndef OPMCONFIG_H
#define OPMCONFIG_H
#include "common.h"

namespace ISfM
{
    struct OpmConfig{
        public:
            EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
            typedef std::shared_ptr<OpmConfig> Ptr;
            bool is_opm_focal = true;
            bool is_opm_principle = true;
            bool is_opm_distortion = true;
            bool is_opm_3d_point = true;
            bool is_opm_pose = true;
    };
}
#endif