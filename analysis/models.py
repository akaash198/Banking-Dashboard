from django.db import models
from django.db.models.signals import post_save,pre_save
from django.dispatch import receiver

class Bank(models.Model):
    name = models.CharField(max_length=100)
    app_id = models.CharField(max_length=100)
    logo = models.ImageField(upload_to='bank_logos/')
    is_active = models.BooleanField(default=False)

    def __str__(self):
        return self.name

class BankReviewData(models.Model):
    bank = models.ForeignKey(Bank, on_delete=models.CASCADE)
    csv_file = models.FileField(upload_to='bank_review_data/')
    created_at = models.DateTimeField(auto_now_add=True)
    is_active = models.BooleanField(default=False)

@receiver(post_save, sender=Bank)
def update_related_review_data(sender, instance, **kwargs):
    # Update related BankReviewData objects when Bank status changes
    related_reviews = BankReviewData.objects.filter(bank=instance)
    related_reviews.update(is_active=instance.is_active)

@receiver(pre_save, sender=Bank)
def update_related_review_data(sender, instance, **kwargs):
    # Update related BankReviewData objects when Bank status changes
    related_reviews = BankReviewData.objects.filter(bank=instance)
    related_reviews.update(is_active=instance.is_active)

class TechnicalAnalysisData(models.Model):
    bank = models.ForeignKey(Bank, on_delete=models.CASCADE)
    analysis_file = models.FileField(upload_to='technical_analysis_data/')
    created_at = models.DateTimeField(auto_now_add=True)
    is_active = models.BooleanField(default=False)

    def __str__(self):
        return f"Technical Analysis Data for {self.bank.name}"

@receiver(post_save, sender=Bank)
def update_related_analysis_data(sender, instance, **kwargs):
    # Update related TechnicalAnalysisData objects when Bank status changes
    related_analysis_data = TechnicalAnalysisData.objects.filter(bank=instance)
    related_analysis_data.update(is_active=instance.is_active)


@receiver(pre_save, sender=Bank)
def update_related_analysis_data(sender, instance, **kwargs):
    # Update related TechnicalAnalysisData objects when Bank status changes
    related_analysis_data = TechnicalAnalysisData.objects.filter(bank=instance)
    related_analysis_data.update(is_active=instance.is_active)

    
