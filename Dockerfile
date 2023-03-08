FROM python:3.8
USER root
RUN mkdir /app
COPY . /app/
WORKDIR /app/
RUN pip install -r requirements.txt

ENV AIRFLOW_HOME='/app/airflow'
ENV AIRFLOW__CORE__DAGBAG_IMPORT_TIMEOUT=1000
ENV AIRFLOW__CORE__ENABLE_XCOM_PICKLING=True
RUN airflow db init
RUN airflow users create -e indianlokesh100@gmail.com -f Lokesh -l Raj -p admin -r Admin -u admin
RUN chmod 777 start.sh
RUN apt update -y &&  apt install awscli -y 
ENTRYPOINT ["/bin/sh"]
CMD['start.sh']
