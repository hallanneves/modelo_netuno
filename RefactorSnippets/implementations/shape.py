
class Shape(object):
	# Create based on class name:
	# @static
	def factory(type):
		# print('tipo escolhido '+ type)
		#return eval(type + "()")
		if type == "Circle": return circle.Circle()
		if type == "Square": return square.Square()
		assert 0, "Bad shape creation: " + type
	factory = staticmethod(factory)
