#ifndef KZZK_CV_H
#define KZZK_CV_H

#include <string>
#include "types.h"

namespace kzzk_cv {

InferenceResult kzzk_cv(const std::string& modelfile, const std::string& imagefile);

} // namespace kzzk_cv

#endif // KZZK_CV_H
