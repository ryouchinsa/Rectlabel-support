import turicreate as tc

data =  tc.SFrame('annotations.sframe')
train_data =  tc.SFrame('annotations.sframe')
model = tc.object_detector.create(train_data, feature = 'image', max_iterations=100)
model.save('mymodel.model')