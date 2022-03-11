FROM tensorflow/tensorflow:latest-gpu

RUN pip install -U PyYAML scikit-learn tqdm retrying flask flask-expects-json gunicorn pandas

WORKDIR /dkn/
VOLUME /dkn/model/ /dkn/data/
EXPOSE 4242
COPY *.py /dkn/
COPY mind-demo-dkn /dkn/mind-demo-dkn
#COPY model_save/final* /dkn/model/

#ENV FLASK_APP=mind_flask
#CMD [ "python3", "-m" , "flask", "run", "--host=0.0.0.0", "--port=4242"]
EXPOSE 4242
ENTRYPOINT ["gunicorn", "-b", "0.0.0.0:4242"]
CMD ["mind_flask:make_app(saved_model='final')"]
