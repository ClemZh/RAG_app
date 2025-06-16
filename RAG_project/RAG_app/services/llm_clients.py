import ollama
from langchain_community.llms import Ollama
import time

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

SELECTED_MODELS = None

@csrf_exempt
def get_available_models(request):
    try:
        models = ollama.list().get("models", [])
        #print(models)
        #print(models[1].model)
        model_names = [model_el.model for model_el in models]
        return JsonResponse({"models": model_names})
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)




# class LLMService:
#     def __ini__(self, model_name)