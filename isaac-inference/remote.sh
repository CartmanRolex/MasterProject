#!/bin/bash

# Configuration pour le streaming à distance
export LIVESTREAM=2
export ENABLE_CAMERAS=1
# On lance le script Python passé en argument
# "$@" permet de passer tous les arguments (ex: --env so101_pick_orange)
python "$@"
