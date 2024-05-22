wget -O- http://neuro.debian.net/lists/bionic.us-nh.full | tee /etc/apt/sources.list.d/neurodebian.sources.list \
 && export GNUPGHOME="$(mktemp -d)" \
 && echo "disable-ipv6" >> ${GNUPGHOME}/dirmngr.conf \
 && (apt-key adv --homedir $GNUPGHOME --recv-keys --keyserver hkp://pgpkeys.eu 0xA5D32F012649A5A9 \
|| { curl -sSL http://neuro.debian.net/_static/neuro.debian.net.asc | apt-key add -; } ) \
 && apt-get update \
 && apt-get install git-annex-standalone git
pip install --no-cache-dir nilearn datalad datalad-osf 
git config --global user.email "dexter.nicholas@gmail.com" && git config --global user.name "ndexter"

datalad clone https://github.com/neuronets/trained-models && \
  cd trained-models && git-annex enableremote osf-storage && \
  datalad get -r -s osf-storage neuronets/braingen/0.1.0/
