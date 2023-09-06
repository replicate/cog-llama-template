if [ ! -d "../official-models" ]; then
    pushd ..
    git clone git@github.com:replicate/official-models
    popd
fi

cp ../official-models/model_secrets/llama-2-13b/.env models/llama-2-13b/
cp ../official-models/model_secrets/llama-2-13b-chat/.env models/llama-2-13b-chat/
cp ../official-models/model_secrets/llama-2-70b/.env models/llama-2-70b/
cp ../official-models/model_secrets/llama-2-70b-chat/.env models/llama-2-70b-chat/
cp ../official-models/model_secrets/llama-2-7b/.env models/llama-2-7b/
cp ../official-models/model_secrets/llama-2-7b-chat/.env models/llama-2-7b-chat/
