from django.contrib import admin

from credit.models import Application, Batch, BehaviourFeatures, Decision


@admin.register(Decision)
class DecisionAdmin(admin.ModelAdmin):
    list_display = ("application", "band", "probability", "model_version", "created_at")

    def has_add_permission(self, request):
        return False

    def has_change_permission(self, request, obj=None):
        return False

    def has_delete_permission(self, request, obj=None):
        return False


admin.site.register(Application)
admin.site.register(BehaviourFeatures)
admin.site.register(Batch)