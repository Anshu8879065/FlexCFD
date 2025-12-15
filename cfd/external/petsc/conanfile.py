from conan import ConanFile
from conan.tools.system import package_manager
from conan.tools.gnu import PkgConfig
from conan.errors import ConanInvalidConfiguration

required_conan_version = ">=2.0"

class PetscSystemPackage(ConanFile):
    name = "petsc"
    version = "system"
    description = "PETSc is a suite of data structures and routines for the scalable (parallel) solution of scientific applications modelled by partial differential equations."
    topics = ("Scientific Computing", "C", "C++", "Fortran")
    homepage = "https://petsc.org/release/"
    license = "BSD Clause-2 'Simplified' License"
    package_type = "shared-library"
    settings = "os", "arch", "compiler", "build_type"

    def package_id(self):
        self.info.clear()

    def validate(self):
        supported_os = ["Linux", "Macos"]
        if self.settings.os not in supported_os:
            raise ConanInvalidConfiguration(f"{self.ref} wraps a system package only supported by {supported_os}.")

    def system_requirements(self):
        apt = package_manager.Apt(self)
        apt.install(["petsc-dev"], update=True, check=True)

        yum = package_manager.Yum(self)
        yum.install(["petsc-mpich-devel"], update=True, check=True)

        brew = package_manager.Brew(self)
        brew.install(["petsc"], update=True, check=True)

    def package_info(self):
        self.cpp_info.binddirs = []
        self.cpp_info.includedirs = []
        self.cpp_info.libdirs = []

        self.cpp_info.set_property("cmake_file_name", "PETSc")
        self.cpp_info.set_property("cmake_target_name", "PETSc::PETSc")
        self.cpp_info.set_property("cmake_additional_variable_prefixes", ["PETSc"])

        pkg_config = PkgConfig(self, "petsc")
        pkg_config.fill_cpp_info(self.cpp_info, is_system=True)
