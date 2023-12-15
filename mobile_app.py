from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.camera import Camera
from kivy.uix.image import Image
from kivy.uix.label import Label
from kivy.core.image import Image as CoreImage
from kivy.clock import Clock
import cv2
import numpy as np


from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#for test time augmentation
test_datagen = ImageDataGenerator(  rescale = 1./255,
                                    horizontal_flip=True,
                                    rotation_range = 20,
                                   width_shift_range = 0.2,
                                   height_shift_range = 0.2,
                                   featurewise_center=True,
                                   featurewise_std_normalization=True,
                                   vertical_flip=True,                                   
                                   zoom_range = 0.2,
                                   )

class DeepLearningApp(App):
    def build(self):
        self.layout = BoxLayout(orientation='vertical')


        self.camera = Camera(resolution=(640, 480), play=True)
        self.layout.add_widget(self.camera)

   
        self.capture_button = Button(text="Capture Photo", size_hint=(None, None), size=(200, 100))
        self.capture_button.bind(on_press=self.capture_photo)
        self.layout.add_widget(self.capture_button)

        self.prediction_label = Label(text='Predicted Class: ')
        self.layout.add_widget(self.prediction_label)

        
        self.captured_image = Image()
        self.layout.add_widget(self.captured_image)

        Clock.schedule_interval(self.update, 1.0 / 30.0)

        return self.layout

    def capture_photo(self, instance):
       
        frame = self.camera.texture.pixels
        frame = np.frombuffer(frame, dtype=np.uint8)
        frame = frame.reshape((self.camera.texture.height, self.camera.texture.width, -1))

       
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        model = load_model('c:/Users/Bhabuk/Desktop/currency mine one/final_model.h5')     
    
        frame = cv2.resize(frame_rgb, (224,224))    
        frame = np.array(frame)        
        frames = np.expand_dims(frame,0)
        input_array = test_datagen.flow(frames,batch_size = 2) 


        
        predictions = model.predict_generator(input_array)
        summed = np.sum(predictions,axis =0)
        predicted_class = np.argmax(summed)
        labels = ['fifty','five','five hundred','hundred','ten','thousand','twenty']
        label = labels[predicted_class]
        self.prediction_label.text = f'Predicted Class: {label}'
  

        

       

    def update(self, dt):
        pass

   
if __name__ == '__main__':
    DeepLearningApp().run()
