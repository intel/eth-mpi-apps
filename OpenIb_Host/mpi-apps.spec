Name: eth-mpi-apps
Version: 11.3.0.0
Release: 130%{?dist}
Summary: Intel MPI benchmarks and Applications used by eth-fast-fabric
Group: System Environment/Libraries
License: BSD and HPL and CPL 1.0
Url: http://www.intel.com/
Source: eth-mpi-apps.tgz

AutoReq: no
Requires: eth-tools-fastfabric%{?_isa} >= %{version}-%{release}
Requires: openblas-devel

BuildRoot: %{_tmppath}/%{name}-%{version}-%{release}-root-%(%{__id_u} -n)

%description
Contains applications and source for testing MPI performance in conjunction with eth-tools-basic or stand alone.

%global debug_package %{nil}

%prep
%setup -q -c


%build



%install

%define mpi_apps_files bandwidth latency hpl-2.3 imb osu-micro-benchmarks-5.9 mpi_multibw mpi_stress


mkdir -p $RPM_BUILD_ROOT/usr/src/eth/mpi_apps

cd ./MpiApps
cp ./apps/* -r $RPM_BUILD_ROOT/usr/src/eth/mpi_apps/

echo "/usr/src/eth/mpi_apps/%{mpi_apps_files}" > %{_builddir}/mpi_apps.list
sed -i 's;[ ];\n/usr/src/eth/mpi_apps/;g' %{_builddir}/mpi_apps.list


%clean
rm -rf $RPM_BUILD_ROOT

%files -f %{_builddir}/mpi_apps.list

%changelog
* Tue Apr 6 2021 Jijun Wang <jijun.wang@intel.com> - 11.1.0.0
- added req on eth-tools-fast-fabric
- removed component information in description

* Mon Feb 26 2018 Jijun Wang <jijun.wang@intel.com> - 10.8.0.0
- Added component information in description for all rpms

* Wed Dec 2 2015 Brandon Yates <brandon.yates@intel.com>
- Initial Creation


