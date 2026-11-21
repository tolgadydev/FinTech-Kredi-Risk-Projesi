from django.core.management.base import BaseCommand

from credit.loading import PREFIXES, load_behaviour


class Command(BaseCommand):
    help = "Ödeme geçmişi özetlerini bir JSONL dosyasından veritabanına yükler."

    def add_arguments(self, parser):
        parser.add_argument("path", help="JSONL dosyasının yolu")
        parser.add_argument("--scope", choices=list(PREFIXES), default="real")

    def handle(self, *args, **options):
        count = load_behaviour(options["path"], options["scope"])
        self.stdout.write(self.style.SUCCESS(f"{count} özet yüklendi ({options['scope']})"))