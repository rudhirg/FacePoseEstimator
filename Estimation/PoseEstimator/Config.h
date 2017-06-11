#ifndef CONFIG_H
#define CONFIG_H

#define TEST_IMAGE "test/test.jpg"
#define MODEL_FILE_PATH "TrainModel.xml"
#define OPTM_ERROR_WEIGHT	(1e-2)//(1.5e-2)

#define FACE_DETECT_MODEL "face_model/haarcascade_frontalface_alt_tree.xml"
#define EYE_DETECT_MODEL "face_model/haarcascade_mcs_eyepair_small.xml"

// params
#define NUM_ITERATONS 40
#define SEARCH_REG_X 32
#define SEARCH_REG_Y 32

#define NUM_PATCHES 58

#define NUM_THREADS 4

#endif