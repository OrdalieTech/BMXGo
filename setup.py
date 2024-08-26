import setuptools

setuptools.setup(
    name="py_BMXGo",
    packages=setuptools.find_packages(include=["py_BMXGo"]),
    py_modules = ["py_BMXGo.BMXGo"],
    package_data={"py_BMXGoOnePackage": ["*.so"]},
    include_package_data=True,
)