#include "base_model.h"

namespace kzzk {

BaseModel::BaseModel() : device_id_(0), initialized_(false), model_path_("") {}

BaseModel::~BaseModel() {}

} // namespace kzzk
