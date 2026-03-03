#!/bin/bash
set -e

# Load CCG_DATA_ROOT from project .env (3 dirs up from this script)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
ENV_FILE="$PROJECT_ROOT/.env"

if [ -f "$ENV_FILE" ]; then
    set -a
    # shellcheck disable=SC1090
    source "$ENV_FILE"
    set +a
fi

if [ -z "$CCG_DATA_ROOT" ]; then
    echo "ERROR: CCG_DATA_ROOT is not set. Set it in $ENV_FILE or your environment." >&2
    exit 1
fi

SOLRING_DIR="$CCG_DATA_ROOT/datasets/solring"
RAW_DIR="$SOLRING_DIR/01_raw"
KEYFRAMES_DIR="$SOLRING_DIR/02_keyframes"

mkdir -p "$KEYFRAMES_DIR"

"$SCRIPT_DIR/split_into_keyframes.sh" "$RAW_DIR/0afa0e33-4804-4b00-b625-c2d6b61090fc_solring_khc_20221219_153056.mp4"
"$SCRIPT_DIR/split_into_keyframes.sh" "$RAW_DIR/58b26011-e103-45c4-a253-900f4e6b2eeb_solring_cmr_20221219_153234.mp4"
"$SCRIPT_DIR/split_into_keyframes.sh" "$RAW_DIR/0f003fde-be17-4159-a361-711ed0bee911_solring_c16_20221219_153333.mp4"
"$SCRIPT_DIR/split_into_keyframes.sh" "$RAW_DIR/71357a3d-9a9f-4ec6-8e01-1966b220206c_solring_cmd_20221219_153427.mp4"
"$SCRIPT_DIR/split_into_keyframes.sh" "$RAW_DIR/199cde21-5bc3-49cd-acd4-bae3af6e5881_solring_clb_20221219_153212.mp4"
"$SCRIPT_DIR/split_into_keyframes.sh" "$RAW_DIR/83a0f2eb-2f6d-4aaa-b7a9-ea06d5de7eca_solring_c18_20221219_153710.mp4"
"$SCRIPT_DIR/split_into_keyframes.sh" "$RAW_DIR/1b3a4537-1d51-47ac-a12e-6b8d68f530e6_solring_mb1_20221219_153504.mp4"
"$SCRIPT_DIR/split_into_keyframes.sh" "$RAW_DIR/8a5edac3-855a-4820-b913-44de5b29b7d0_solring_znc_20221219_153253.mp4"
"$SCRIPT_DIR/split_into_keyframes.sh" "$RAW_DIR/1b59533a-3e38-495d-873e-2f89fbd08494_solring_c13_20221219_153409.mp4"
"$SCRIPT_DIR/split_into_keyframes.sh" "$RAW_DIR/b79cb394-eb91-4b3b-91d4-c6a0f723feb1_solring_c14_20221219_153445.mp4"
"$SCRIPT_DIR/split_into_keyframes.sh" "$RAW_DIR/286bea73-8ad8-4423-8a7c-8497420fdb54_solring_c20_20221219_153135.mp4"
"$SCRIPT_DIR/split_into_keyframes.sh" "$RAW_DIR/beebe533-29b9-4041-ab66-0a8233c50d56_solring_dmc_20221219_153029.mp4"
"$SCRIPT_DIR/split_into_keyframes.sh" "$RAW_DIR/2c52c96d-e20f-4025-b759-674b36cf0db3_solring_afc_20221219_153115.mp4"
"$SCRIPT_DIR/split_into_keyframes.sh" "$RAW_DIR/c6399a22-cebf-4c1d-a23e-4c68f784ac1b_solring_c17_20221219_153312.mp4"
"$SCRIPT_DIR/split_into_keyframes.sh" "$RAW_DIR/3459b229-7c46-4f70-87d4-bb31c2c17dd9_solring_c15_20221219_153523.mp4"
"$SCRIPT_DIR/split_into_keyframes.sh" "$RAW_DIR/e672d408-997c-4a19-810a-3da8411eecf2_solring_c19_20221219_153559.mp4"
"$SCRIPT_DIR/split_into_keyframes.sh" "$RAW_DIR/38d347b7-dc17-417a-ab07-29fe99b9a101_solring_phed_20221219_153646.mp4"
"$SCRIPT_DIR/split_into_keyframes.sh" "$RAW_DIR/f48f7190-9ee3-477f-8b25-91e8c2916624_solring_cma_20221219_153349.mp4"
"$SCRIPT_DIR/split_into_keyframes.sh" "$RAW_DIR/3917f744-b876-47ae-94ad-f72b215ff1e7_solring_nec_20221219_153541.mp4"
"$SCRIPT_DIR/split_into_keyframes.sh" "$RAW_DIR/f9a32f17-49c4-4654-a087-1ba474f37377_solring_cm2_20221219_153152.mp4"
"$SCRIPT_DIR/split_into_keyframes.sh" "$RAW_DIR/4cbc6901-6a4a-4d0a-83ea-7eefa3b35021_solring_c21_20221219_153619.mp4"

mv "$RAW_DIR"/*.jpg "$KEYFRAMES_DIR/"

echo "All done!"
