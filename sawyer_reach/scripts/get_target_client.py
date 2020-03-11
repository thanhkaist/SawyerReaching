import rospy
from sawyer_reach.srv import target, targetResponse, targetRequest

service_name = "/get_target"
service = rospy.ServiceProxy(service_name, target)
service.wait_for_service()
print("Successful connection to '" + service_name + "'.")
req = targetRequest()
req.data = 0
resp = service.call(req)
print(resp)