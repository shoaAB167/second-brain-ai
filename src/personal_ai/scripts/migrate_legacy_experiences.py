"""Utility script to safely associate legacy NULL-user Experiences with an authenticated user."""

import argparse
import asyncio
import uuid
from sqlalchemy import select, update
from personal_ai.db.models import ExperienceModel, User
from personal_ai.db.session import AsyncSessionFactory


async def assign_legacy_experiences(target_email: str = None, target_user_id: uuid.UUID = None) -> int:
    """Safely migrate legacy Experiences with user_id = NULL to a verified authenticated user."""
    async with AsyncSessionFactory() as session:
        # 1. Resolve target user
        if target_user_id:
            user_stmt = select(User).where(User.id == target_user_id)
        elif target_email:
            user_stmt = select(User).where(User.email == target_email)
        else:
            raise ValueError("Must provide either target_email or target_user_id.")

        res = await session.execute(user_stmt)
        user = res.scalar_one_or_none()

        if not user:
            raise ValueError(f"Target user not found (email={target_email}, id={target_user_id})")

        # 2. Find legacy experiences
        exp_stmt = select(ExperienceModel).where(ExperienceModel.user_id.is_(None))
        res_exp = await session.execute(exp_stmt)
        legacy_records = res_exp.scalars().all()

        if not legacy_records:
            print("No legacy experiences with user_id=NULL found.")
            return 0

        print(f"Found {len(legacy_records)} legacy experiences to assign to user: {user.email} ({user.id})")

        # 3. Update records deterministically
        for rec in legacy_records:
            rec.user_id = user.id
            print(f"  Assigned Experience ID: {rec.id} (Content preview: {rec.content[:50]}...)")

        await session.commit()
        print(f"Successfully migrated {len(legacy_records)} legacy experiences.")
        return len(legacy_records)


def main():
    parser = argparse.ArgumentParser(description="Migrate legacy NULL-user Experiences to an authenticated user.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--email", type=str, help="Email of the authenticated user to claim legacy experiences.")
    group.add_argument("--user-id", type=str, help="UUID of the authenticated user to claim legacy experiences.")

    args = parser.parse_args()
    target_uuid = uuid.UUID(args.user_id) if args.user_id else None

    asyncio.run(assign_legacy_experiences(target_email=args.email, target_user_id=target_uuid))


if __name__ == "__main__":
    main()
