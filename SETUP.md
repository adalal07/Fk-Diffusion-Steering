conda create -n fk python=3.10
python -m pip install --no-build-isolation "git+ssh://git@github.com/THUDM/ImageReward.git@2ca71bac4ed86b922fe53ddaec3109fe94d45fd3#egg=image_reward"
python -m pip install --no-build-isolation -r requirements.txt