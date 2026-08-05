#ifndef KZZK_CV_H
#define KZZK_CV_H

#include <string>
#include "types.h"

namespace kzzk {

InferenceResult kzzk_cv(const std::string& modelfile, const std::string& imagefile);

} // namespace kzzk

#endif // KZZK_CV_H
