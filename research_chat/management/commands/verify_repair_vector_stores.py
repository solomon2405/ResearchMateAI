"""
Management command to audit, verify, and repair FAISS vector store consistency.
Audits vector dimensions, counts, mapping files, and database chunk alignment per project.
Safe and idempotent to run repeatedly.
"""

from django.core.management.base import BaseCommand
from research_projects.models import ResearchProject
from research_chat.services.vector_store import get_project_vector_store


class Command(BaseCommand):
    help = "Verify and repair consistency between FAISS vector indexes and PostgreSQL/SQLite database records."

    def add_arguments(self, parser):
        parser.add_argument(
            '--project-id',
            type=int,
            default=None,
            help="Target a specific research project ID."
        )
        parser.add_argument(
            '--repair',
            action='store_true',
            help="Automatically rebuild corrupt, missing, or mismatched vector indexes."
        )

    def handle(self, *args, **options):
        project_id = options.get('project_id')
        repair_mode = options.get('repair', False)

        if project_id:
            projects = ResearchProject.objects.filter(pk=project_id)
            if not projects.exists():
                self.stderr.write(self.style.ERROR(f"Project with ID {project_id} does not exist."))
                return
        else:
            projects = ResearchProject.objects.all().order_by('id')

        total = projects.count()
        self.stdout.write(self.style.MIGRATE_HEADING(
            f"Auditing FAISS vector stores for {total} research project(s)... "
            f"({'Auto-Repair: ON' if repair_mode else 'Check Only'})"
        ))

        consistent_count = 0
        inconsistent_count = 0
        repaired_count = 0

        for project in projects:
            store = get_project_vector_store(project.id)
            audit = store.validate_integrity()

            if audit['is_valid']:
                consistent_count += 1
                self.stdout.write(self.style.SUCCESS(
                    f"  [OK] Project #{project.id} ('{project.name[:30]}'): Healthy "
                    f"({audit['vector_count']} vectors, {audit['db_eligible_count']} DB chunks)"
                ))
            else:
                inconsistent_count += 1
                self.stdout.write(self.style.WARNING(
                    f"  [!] Project #{project.id} ('{project.name[:30]}'): INCONSISTENT -- {audit.get('issue')}"
                ))

                if repair_mode:
                    self.stdout.write(f"    Rebuilding index for project #{project.id}...")
                    try:
                        result = store.rebuild_from_database()
                        repaired_count += 1
                        self.stdout.write(self.style.SUCCESS(
                            f"    [OK] Successfully repaired project #{project.id} ({result['indexed_chunks']} chunks indexed)."
                        ))
                    except Exception as e:
                        self.stderr.write(self.style.ERROR(f"    Failed to repair project #{project.id}: {e}"))

        self.stdout.write("\n" + self.style.MIGRATE_LABEL("Summary Report:"))
        self.stdout.write(f"  Total Projects Checked: {total}")
        self.stdout.write(f"  Healthy:                {consistent_count}")
        self.stdout.write(f"  Inconsistent:           {inconsistent_count}")
        if repair_mode:
            self.stdout.write(f"  Repaired:               {repaired_count}")
        elif inconsistent_count > 0:
            self.stdout.write(self.style.NOTICE("  Tip: Run with --repair to automatically restore broken indexes."))
