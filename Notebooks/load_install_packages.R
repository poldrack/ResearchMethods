load_install_package = function(package, apt=TRUE) {
  if (!require(package, character.only=T, quietly=T)) {
    if (apt) {
        apt_install(package)
    } else {
        install.packages(package)
    }
    library(package, character.only=T)
  }
}

apt_install = function(package){
  system2(command='apt-get',
          args=c('install', sprintf('r-cran-%s', tolower(package))),
          stderr=TRUE, stdout=TRUE)
}

