from django.contrib import admin
from .models import Bank, BankReviewData, TechnicalAnalysisData

class BankAdmin(admin.ModelAdmin):
    list_display = ('name', 'app_id', 'is_active')
    list_filter = ('is_active',)
    search_fields = ('name', 'app_id')
    ordering = ('name',)
    actions = ['set_selected_bank_active', 'set_selected_bank_inactive']

    def set_selected_bank_active(self, request, queryset):
        # Activate selected banks
        selected_banks = queryset.update(is_active=True)

        # Deactivate all other banks
        Bank.objects.exclude(pk__in=queryset.values_list('pk', flat=True)).update(is_active=False)

        # Update related BankReviewData objects for selected banks
        BankReviewData.objects.filter(bank__in=queryset).update(is_active=True)

        # Update related TechnicalAnalysisData objects for selected banks
        TechnicalAnalysisData.objects.filter(bank__in=queryset).update(is_active=True)

        # Deactivate related BankReviewData objects for banks that are not in the selected queryset
        BankReviewData.objects.exclude(bank__in=queryset).update(is_active=False)

        # Deactivate related TechnicalAnalysisData objects for banks that are not in the selected queryset
        TechnicalAnalysisData.objects.exclude(bank__in=queryset).update(is_active=False)

        self.message_user(request, f'Selected bank(s) set as active: {selected_banks}')

    set_selected_bank_active.short_description = 'Set selected bank(s) as active'

    def set_selected_bank_inactive(self, request, queryset):
        # Deactivate selected banks
        selected_banks = queryset.update(is_active=False)

        # Update related BankReviewData objects for selected banks
        BankReviewData.objects.filter(bank__in=queryset).update(is_active=False)

        # Update related TechnicalAnalysisData objects for selected banks
        TechnicalAnalysisData.objects.filter(bank__in=queryset).update(is_active=False)

        self.message_user(request, f'Selected bank(s) set as inactive: {selected_banks}')

    set_selected_bank_inactive.short_description = 'Set selected bank(s) as inactive'

    def save_model(self, request, obj, form, change):
        # Save the Bank object
        obj.save()

        # Update related BankReviewData objects
        related_reviews = BankReviewData.objects.filter(bank=obj)
        related_reviews.update(is_active=obj.is_active)

        # Update related TechnicalAnalysisData objects
        related_analysis = TechnicalAnalysisData.objects.filter(bank=obj)
        related_analysis.update(is_active=obj.is_active)

admin.site.register(Bank, BankAdmin)

class BankReviewDataAdmin(admin.ModelAdmin):
    list_display = ('bank', 'created_at', 'is_active')
    list_filter = ('bank', 'is_active')
    search_fields = ('bank',)
    ordering = ('bank',)
    actions = ['activate_bank_review_data', 'deactivate_bank_review_data']

    def activate_bank_review_data(self, request, queryset):
        # Activate selected bank review data
        selected_bank_review_data = queryset.update(is_active=True)

        self.message_user(request, f'Selected bank review data(s) set as active: {selected_bank_review_data}')

    activate_bank_review_data.short_description = 'Set selected bank review data(s) as active'

    def deactivate_bank_review_data(self, request, queryset):
        # Deactivate selected bank review data
        selected_bank_review_data = queryset.update(is_active=False)

        self.message_user(request, f'Selected bank review data(s) set as inactive: {selected_bank_review_data}')

admin.site.register(BankReviewData, BankReviewDataAdmin)

class TechnicalAnalysisDataAdmin(admin.ModelAdmin):
    list_display = ('bank', 'created_at', 'is_active')
    list_filter = ('bank', 'is_active')
    search_fields = ('bank',)
    ordering = ('bank',)
    actions = ['activate_bank_analysis_data', 'deactivate_bank_analysis_data']

    def activate_bank_analysis_data(self, request, queryset):
        # Activate selected bank analysis data
        selected_bank_analysis_data = queryset.update(is_active=True)

        self.message_user(request, f'Selected bank analysis data(s) set as active: {selected_bank_analysis_data}')

    activate_bank_analysis_data.short_description = 'Set selected bank analysis data(s) as active'

    def deactivate_bank_analysis_data(self, request, queryset):
        # Deactivate selected bank analysis data
        selected_bank_analysis_data = queryset.update(is_active=False)

        self.message_user(request, f'Selected bank analysis data(s) set as inactive: {selected_bank_analysis_data}')

admin.site.register(TechnicalAnalysisData, TechnicalAnalysisDataAdmin)
