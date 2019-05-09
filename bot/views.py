from django.shortcuts import render
from django.template.context_processors import csrf
from entity_model.views import get_unique_entity
from entity_model.views import predict_entity
from intent_model.views import predict_intent
from intent_model.views import train_intent_model
from entity_model.views import train_entity_model
from intent_model.views import get_unique_intent
from .models import UniqueEntity,UniqueIntent,Responses

# Create your views here.

chatt = []
train_intent_model()
train_entity_model()


def index(request):
    return render(request, "home1.html")


def process(request):
    c = {}
    c.update(csrf(request))
    # client_ip = request.environ.get('REMOTE_ADDR')
    # request.session['user'] = client_ip
    msg = request.POST.get('msg', '')
    entity=predict_entity(msg, get_unique_entity())
    intent = predict_intent(msg, get_unique_intent())
    chatt.append(msg)
    for ent in UniqueEntity.objects.all():
        if ent.Entity == entity:
            entid = ent.EId
    for it in UniqueIntent.objects.all():
        if it.Intent == intent:
            intid = it.IId
    for resp in Responses.objects.all():
        if resp.Entity_id.EId == entid and resp.Intent_id.IId == intid:
            chatt.append(resp.Response)
            break
    context = {'chat': chatt, 'isopen': 'true'}
    return render(request, "home1.html", context)
