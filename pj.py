import phonenumbers
from phonenumbers import geocoder
ph_no1=phonenumbers.parse("+917627069858")
print(geocoder.description_for_number(ph_no1,"en"))