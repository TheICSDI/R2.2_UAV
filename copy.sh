cp /tests/case_studies /src/aerialist -r
cp /tests/middle.py /src/aerialist
cp /tests/testcase.py /src/aerialist
export PYTHONPATH=/src/aerialist:$PYTHONPATH
python3 /src/aerialist/middle.py
