export CUDA_VISIBLE_DEVICES=0

# DATA_ROOT_DIR="/mnt/nas-a6000/hanyangyu/projects/Dataset/tanks360/"
DATA_ROOT_DIR="./data"

SCENES=(
    horse
    # family360
    # ignatius360
    # trunk360
    )

Prompts=(
    'a bronze statue of a rearing horse prominently displayed on a pedestal in a landscaped park, surrounded by palm trees and modern architecture, with people strolling in the background.'
    # 'a bronze statue depicting a tender interaction between a mother holding a child and a kneeling figure, set in a landscaped courtyard with benches and palm trees.'
    # 'a serene park with a central bronze statue of a figure holding a child, surrounded by lush greenery, flowering borders, and elegant buildings in the background.'
    # 'a vintage turquoise truck with a wooden flatbed parked on a city sidewalk, surrounded by modern buildings and construction barriers.'
)
N_VIEWS=(
    # 2
    # 3
    # 4
    16
    )

for i in "${!SCENES[@]}"; do
    SCENE=${SCENES[$i]}
    PROMPT=${Prompts[$i]}

    for N_VIEW in "${N_VIEWS[@]}"; do

        test_iterations1=6000
        test_iterations2=30000

        dataset=${DATA_ROOT_DIR}/${SCENE}_${N_VIEW}
        workspace="./output/${SCENE}_${N_VIEW}"
        model_load="./output/${SCENE}_${N_VIEW}/chkpnt${test_iterations1}.pth"
        model_load2="./output/${SCENE}_${N_VIEW}/chkpnt${test_iterations2}.pth"
        video_path="./output/${SCENE}_${N_VIEW}/${test_iterations2}_render_video.mp4"

        echo "Model load path: ${model_load}"
        mkdir -p ${workspace}
        echo "======================================================="
        echo "Starting process: ${SCENE} (${N_VIEW} views/${gs_train_iter} iters) on GPU ${GPU_ID}"
        echo "======================================================="



        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting Dense Initialization..."
        python dust3r/coarse_initialization.py -s $dataset \
        > ${workspace}/01_init.log 2>&1
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Dense Initialization completed. Log saved in ${workspace}/01_init.log"



        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting getting monocular depth/normal..."        
        python Marigold/getmonodepthnormal.py -s $dataset \
        > ${workspace}/01_mono.log 2>&1
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Monocular estimation completed. Log saved in ${workspace}/01_mono.log"



        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Stage1..."        
        python stage1_360.py  -s $dataset --save $workspace -r 2 --checkpoint_iterations ${test_iterations1} \
        > ${workspace}/02_stage_1.log 2>&1
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Stage_1 completed. Log saved in ${workspace}/02_stage_1.log"



        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Train repair model..." 
        python train_repairmodel.py   --exp_name output/repairmodel/${SCENE}_${N_VIEW} \
        --prompt "$PROMPT" \
        --resolution 1 --gs_dir  $workspace --data_dir $dataset   --bg_white \
        > ${workspace}/03_repair.log 2>&1
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Repair model training completed. Log saved in ${workspace}/03_repair.log"



        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Stage2..." 
        python stage2_360.py  -s $dataset  -r 2 --exp_name output/repairmodel/${SCENE}_${N_VIEW} \
        --prompt "$PROMPT" \
        --bg_white  --start_checkpoint $model_load \
        > ${workspace}/04_stage2.log 2>&1
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Stage_2 completed. Log saved in ${workspace}/04_stage2.log"

        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Interpolate images..." 
        python render_interpolate.py  -s $dataset   --start_checkpoint $model_load2
        > ${workspace}/05_interpolate.log 2>&1
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Interpolate completed. Log saved in ${workspace}/05_interpolate.log"

        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Scene enhance..." 
        python scene_enhance.py  --model_path /mnt/nas-a6000/hanyangyu/projects/lm-gaussian/models/zeroscope_v2_XL   --input_path ${video_path}
        > ${workspace}/06_scene_enhance.log 2>&1
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Scene enhance. Log saved in ${workspace}/06_scene_enhance.log"
    done
done