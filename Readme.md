1. Odroid ubuntu 16.04 ��ġ (X86�� ��� ubuntu 16.04 + GPU driver + opencl sdk ��ġ)
2. Dependency Library ��ġ 
  apt-get install libffi-dev python-dev python-pip python-numpy python-mako python-yaml protobuf-compiler libprotobuf-dev libboost-python-dev python-opencv python-matplotlib python-scipy
  pip install pyopencl 
  pip install enum
  pip install protobuf
  pip install easydict
3. protobuf compile
  ./build.sh
4. ����
  Simple demo: python test.py
  MNIST demo: python mnist.py
  Faster-RCNN demo: python demo.py 
  

Notice. 
1. dependency�� ������Ʈ �Ǿ����ϴ�. ���� ���̺귯�� �ܿ� �߰��� ���̺귯������ ������ �ش� ���̺귯���� ��ġ�ϼž� �մϴ�.
2. proto read ������ ���Ǿ����ϴ�.
3. ���� python framework ���� �κ� �Ϻΰ� ����Ǿ����ϴ�. 
4. Odroid���� Faster-RCNN ������ �����մϴ�. ��, ������ �ڵ� ������ ���װ� �����Ͽ� ������ shell���� 2�� ������ �Ұ����մϴ�. �ٽ� demo�� �����Ϸ��� �ٸ� shell�� �����Ͻð� �����Ű�� �˴ϴ�.
5. py-faster-rcnn���� �����ϴ� ���۷��� ��Ʈ��ũ�� �Բ� �����˴ϴ�. VOC2007 test set���� 59.31 %�� mAP�� �����ϴ�.
  
 
 
 