FROM reg.docker.alibaba-inc.com/atorch/atorch-dev:20230808torch210dev20230731cu118nlp

USER root
WORKDIR /root

ENV BASH_ENV /root/.bashrc
ENV LANGUAGE zh_cn
ENV LC_ALL zh_CN.UTF-8
ENV SHELL /bin/bash
SHELL ["/bin/bash","-c"]

ADD lib /root/builder

RUN rm -rf /pai-extension && mv ~/builder/pai-extension /pai-extension && chmod 777 -R /pai-extension
RUN mv ~/builder/theia-ide/.theia ~/.theia && rm -rf ~/.aistudio/hooks/*

RUN sh ~/builder/script/install-dumb-init.sh

RUN sh ~/builder/script/install-node.sh v5.20.3 \
    && rm -rf ~/.aistudio && mkdir -p ~/.aistudio \
    && echo 'export npm_config_user=root' >> ~/.bashrc \
    && mv ~/builder/theia-ide/.aistudio/* ~/.aistudio \
    && mv ~/builder/theia-ide/.aistudio/.[^.]* ~/.aistudio \
    && gcc --version

RUN sh ~/builder/script/install-third-common.sh
RUN sh ~/builder/script/python/install-sdk.sh || echo "install sdk failed"
RUN sh ~/builder/script/matplot/installer.sh
RUN pip install -I urllib3==1.26.4 && sh ~/builder/script/python/install-jupyter.sh
RUN sh ~/builder/script/setup-base.sh
RUN pip install jinja2==2.11.3 --no-deps
RUN pip install markupsafe==1.1.1 --no-deps

# git lfs
RUN pip install gradio==3.20.1

RUN npm i -g @alipay/aistudio-bootstrap \
    && npm i -g @alipay/aistudio-installer-cli \
    && ais-installer install full \
    && ais-installer collect --version=${IMAGEVERSION} --type=${IMAGETYPE}

RUN pip install -U transformers==4.30.1
RUN pip install -U bitsandbytes==0.39.0
RUN pip install -U accelerate==0.20.3
RUN pip install peft==0.4.0
RUN pip uninstall flash_attn -y
RUN pip install xformers --no-deps
RUN pip install -U atorch==0.1.7rc17 --no-deps
RUN pip install zstandard
RUN pip install ujson
RUN pip install jsonlines