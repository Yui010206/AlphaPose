set -x

CONFIG=$1
EXPID=${2:-"alphapose"}

python ./scripts/train_for_2stage.py \
    --exp-id ${EXPID} \
    --cfg ${CONFIG}
