import os
import sib_api_v3_sdk
from sib_api_v3_sdk.rest import ApiException
from datetime import datetime, timedelta
from fastapi import APIRouter, HTTPException, Response, Request, Depends
from pydantic import BaseModel, EmailStr
from passlib.context import CryptContext
from jose import JWTError, jwt
from pymongo import MongoClient
from bson import ObjectId
import secrets
import re

# Database connection
mongo_uri = os.environ["MONGO_URI"]
mongo_client = MongoClient(mongo_uri)
db = mongo_client["mozaic_db"]
users_collection = db["users"]
verification_tokens_collection = db["email_verification_tokens"]
two_fa_tokens_collection = db["two_fa_tokens"]

# Password hashing setup
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# JWT setup
SECRET_KEY = os.environ["JWT_SECRET"]
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 15
REFRESH_TOKEN_EXPIRE_DAYS = 7

# Router setup
router = APIRouter(prefix="/auth", tags=["Authentication"])

# -------------------- HELPER FUNCTIONS --------------------

def hash_password(password: str) -> str:
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def validate_password(password: str):
    if len(password) < 8:
        raise HTTPException(status_code=400, detail="Password must be at least 8 characters")
    if not re.search(r"[A-Z]", password):
        raise HTTPException(status_code=400, detail="Password must contain at least one capital letter")
    if not re.search(r"[a-z]", password):
        raise HTTPException(status_code=400, detail="Password must contain at least one small letter")
    if not re.search(r"[0-9]", password):
        raise HTTPException(status_code=400, detail="Password must contain at least one number")
    if not re.search(r"[!@#$%^&*(),.?\":{}|<>]", password):
        raise HTTPException(status_code=400, detail="Password must contain at least one special character")

def create_access_token(user_id: str) -> str:
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    payload = {"sub": user_id, "exp": expire, "type": "access"}
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

def create_refresh_token(user_id: str) -> str:
    expire = datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    payload = {"sub": user_id, "exp": expire, "type": "refresh"}
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

def verify_token(token: str, token_type: str) -> str:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        if payload.get("type") != token_type:
            raise HTTPException(status_code=401, detail="Invalid token type")
        user_id = payload.get("sub")
        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid token")
        return user_id
    except JWTError:
        raise HTTPException(status_code=401, detail="Token is invalid or expired")

# -------------------- REQUEST MODELS --------------------

class RegisterRequest(BaseModel):
    firstName: str
    lastName: str
    email: EmailStr
    phoneNumber: str
    password: str

# -------------------- REGISTER ENDPOINT --------------------

@router.post("/register")
def register(body: RegisterRequest, response: Response):
    try:
        validate_password(body.password)

        existing_user = users_collection.find_one({"email": body.email})
        if existing_user:
            raise HTTPException(
                status_code=400,
                detail="This email is already registered. Please login instead."
            )

        hashed_password = hash_password(body.password)

        new_user = {
            "firstName": body.firstName,
            "lastName": body.lastName,
            "email": body.email,
            "phoneNumber": body.phoneNumber,
            "passwordHash": hashed_password,
            "bio": "",
            "isEmailVerified": False,
            "isTwoFAEnabled": False,
            "twoFAMethod": "email",
            "failedLoginAttempts": 0,
            "lockUntil": None,
            "createdAt": datetime.utcnow()
        }

        result = users_collection.insert_one(new_user)
        user_id = str(result.inserted_id)

        verification_token = secrets.token_urlsafe(32)

        verification_tokens_collection.insert_one({
            "userId": user_id,
            "token": verification_token,
            "expiresAt": datetime.utcnow() + timedelta(hours=24),
            "used": False
        })

        verification_link = f"{os.environ['FRONTEND_URL']}/verify-email?token={verification_token}"
        configuration = sib_api_v3_sdk.Configuration()
        configuration.api_key['api-key'] = os.environ["BREVO_API_KEY"]
        api_instance = sib_api_v3_sdk.TransactionalEmailsApi(sib_api_v3_sdk.ApiClient(configuration))
        send_smtp_email = sib_api_v3_sdk.SendSmtpEmail(
            to=[{"email": body.email, "name": body.firstName}],
            sender={"name": "MozaicTeck", "email": os.environ["BREVO_SENDER_EMAIL"]},
            subject="Verify your MozaicTeck account",
            html_content=f"""
                <h2>Welcome to MozaicTeck, {body.firstName}!</h2>
                <p>Thank you for registering. Please click the link below to verify your email address.</p>
                <a href="{verification_link}" style="background-color:#E8650A;color:white;padding:12px 24px;text-decoration:none;border-radius:6px;">
                    Verify My Email
                </a>
                <p>This link expires in 24 hours.</p>
                <p>If you did not create this account, please ignore this email.</p>
            """
        )
        api_instance.send_transac_email(send_smtp_email)

        return {
            "message": "Registration successful. Please check your email to verify your account."
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail="Something went wrong during registration. Please try again."
        )

# -------------------- VERIFY EMAIL ENDPOINT --------------------

@router.get("/verify-email")
def verify_email(token: str):
    try:
        token_record = verification_tokens_collection.find_one({"token": token})

        if not token_record:
            raise HTTPException(
                status_code=400,
                detail="Invalid verification link. Please register again."
            )

        if token_record["used"]:
            raise HTTPException(
                status_code=400,
                detail="This verification link has already been used. Please login."
            )

        if datetime.utcnow() > token_record["expiresAt"]:
            raise HTTPException(
                status_code=400,
                detail="This verification link has expired. Please register again."
            )

        users_collection.update_one(
            {"_id": ObjectId(token_record["userId"])},
            {"$set": {"isEmailVerified": True}}
        )

        verification_tokens_collection.update_one(
            {"token": token},
            {"$set": {"used": True}}
        )

        return {
            "message": "Email verified successfully. You can now login to your account."
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail="Something went wrong during verification. Please try again."
        )

# -------------------- REQUEST MODELS --------------------

class LoginRequest(BaseModel):
    email: EmailStr
    password: str

# -------------------- LOGIN ENDPOINT --------------------

@router.post("/login")
def login(body: LoginRequest, response: Response):
    try:
        user = users_collection.find_one({"email": body.email})
        if not user:
            raise HTTPException(
                status_code=401,
                detail="Invalid email or password."
            )

        if user.get("lockUntil") and datetime.utcnow() < user["lockUntil"]:
            remaining = int((user["lockUntil"] - datetime.utcnow()).total_seconds())
            raise HTTPException(
                status_code=429,
                detail=f"Account temporarily locked. Please try again in {remaining} seconds."
            )

        if not user["isEmailVerified"]:
            raise HTTPException(
                status_code=401,
                detail="Please verify your email address before logging in."
            )

        if not verify_password(body.password, user["passwordHash"]):
            failed_attempts = user.get("failedLoginAttempts", 0) + 1

            if failed_attempts >= 5:
                users_collection.update_one(
                    {"email": body.email},
                    {"$set": {
                        "failedLoginAttempts": failed_attempts,
                        "lockUntil": datetime.utcnow() + timedelta(seconds=60)
                    }}
                )
                raise HTTPException(
                    status_code=429,
                    detail="Too many failed attempts. Account locked for 60 seconds."
                )
            else:
                users_collection.update_one(
                    {"email": body.email},
                    {"$set": {"failedLoginAttempts": failed_attempts}}
                )
                raise HTTPException(
                    status_code=401,
                    detail=f"Invalid email or password. {5 - failed_attempts} attempts remaining."
                )

        users_collection.update_one(
            {"email": body.email},
            {"$set": {
                "failedLoginAttempts": 0,
                "lockUntil": None
            }}
        )

        user_id = str(user["_id"])
        access_token = create_access_token(user_id)
        refresh_token = create_refresh_token(user_id)

        response.set_cookie(
            key="access_token",
            value=access_token,
            httponly=True,
            samesite="none",
            secure=True,
            max_age=15 * 60
        )
        response.set_cookie(
            key="refresh_token",
            value=refresh_token,
            httponly=True,
            samesite="lax",
            secure=True,
            max_age=7 * 24 * 60 * 60
        )

        return {
            "message": "Login successful.",
            "user": {
                "firstName": user["firstName"],
                "lastName": user["lastName"],
                "email": user["email"]
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail="Something went wrong during login. Please try again."
        )

# -------------------- LOGOUT ENDPOINT --------------------

@router.post("/logout")
def logout(response: Response):
    try:
        response.delete_cookie(
            key="access_token",
            httponly=True,
            samesite="lax",
            secure=True
        )

        response.delete_cookie(
            key="refresh_token",
            httponly=True,
            samesite="lax",
            secure=True
        )

        return {
            "message": "Logged out successfully."
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail="Something went wrong during logout. Please try again."
        )

# -------------------- REFRESH TOKEN ENDPOINT --------------------

@router.post("/refresh")
def refresh_token(request: Request, response: Response):
    try:
        token = request.cookies.get("refresh_token")
        if not token:
            raise HTTPException(
                status_code=401,
                detail="No refresh token found. Please login again."
            )

        user_id = verify_token(token, "refresh")

        user = users_collection.find_one({"_id": ObjectId(user_id)})
        if not user:
            raise HTTPException(
                status_code=401,
                detail="User no longer exists. Please login again."
            )

        new_access_token = create_access_token(user_id)

        response.set_cookie(
            key="access_token",
            value=new_access_token,
            httponly=True,
            samesite="lax",
            secure=True,
            max_age=15 * 60
        )

        return {
            "message": "Token refreshed successfully."
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail="Something went wrong. Please login again."
        )

# -------------------- REQUEST MODELS --------------------

class ForgotPasswordRequest(BaseModel):
    email: EmailStr

# -------------------- FORGOT PASSWORD ENDPOINT --------------------

@router.post("/forgot-password")
def forgot_password(body: ForgotPasswordRequest):
    try:
        user = users_collection.find_one({"email": body.email})

        if user:
            reset_token = secrets.token_urlsafe(32)

            verification_tokens_collection.insert_one({
                "userId": str(user["_id"]),
                "token": reset_token,
                "type": "password_reset",
                "expiresAt": datetime.utcnow() + timedelta(minutes=15),
                "used": False
            })

            reset_link = f"{os.environ['FRONTEND_URL']}/reset-password?token={reset_token}"
            configuration = sib_api_v3_sdk.Configuration()
            configuration.api_key['api-key'] = os.environ["BREVO_API_KEY"]
            api_instance = sib_api_v3_sdk.TransactionalEmailsApi(sib_api_v3_sdk.ApiClient(configuration))
            send_smtp_email = sib_api_v3_sdk.SendSmtpEmail(
                to=[{"email": body.email, "name": user['firstName']}],
                sender={"name": "MozaicTeck", "email": os.environ["BREVO_SENDER_EMAIL"]},
                subject="Reset your MozaicTeck password",
                html_content=f"""
                    <h2>Password Reset Request</h2>
                    <p>Hi {user['firstName']},</p>
                    <p>We received a request to reset your password.
                    Click the button below to create a new password.</p>
                    <a href="{reset_link}" style="background-color:#E8650A;color:white;padding:12px 24px;text-decoration:none;border-radius:6px;">
                        Reset My Password
                    </a>
                    <p>This link expires in 15 minutes.</p>
                    <p>If you did not request a password reset,
                    please ignore this email. Your password will not change.</p>
                """
            )
            api_instance.send_transac_email(send_smtp_email)

        return {
            "message": "If this email is registered you will receive a password reset link shortly."
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail="Something went wrong. Please try again."
        )

# -------------------- REQUEST MODELS --------------------

class ResetPasswordRequest(BaseModel):
    token: str
    new_password: str

# -------------------- RESET PASSWORD ENDPOINT --------------------

@router.post("/reset-password")
def reset_password(body: ResetPasswordRequest):
    try:
        token_record = verification_tokens_collection.find_one({
            "token": body.token,
            "type": "password_reset"
        })

        if not token_record:
            raise HTTPException(
                status_code=400,
                detail="Invalid reset link. Please request a new one."
            )

        if token_record["used"]:
            raise HTTPException(
                status_code=400,
                detail="This reset link has already been used. Please request a new one."
            )

        if datetime.utcnow() > token_record["expiresAt"]:
            raise HTTPException(
                status_code=400,
                detail="This reset link has expired. Please request a new one."
            )

        validate_password(body.new_password)

        new_hashed_password = hash_password(body.new_password)

        users_collection.update_one(
            {"_id": ObjectId(token_record["userId"])},
            {"$set": {
                "passwordHash": new_hashed_password,
                "failedLoginAttempts": 0,
                "lockUntil": None
            }}
        )

        verification_tokens_collection.update_one(
            {"token": body.token},
            {"$set": {"used": True}}
        )

        return {
            "message": "Password reset successful. You can now login with your new password."
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail="Something went wrong. Please try again."
        )

# -------------------- GET CURRENT USER ENDPOINT --------------------

@router.get("/me")
def get_current_user(request: Request):
    try:
        token = request.cookies.get("access_token")
        if not token:
            raise HTTPException(
                status_code=401,
                detail="Not authenticated. Please login."
            )

        user_id = verify_token(token, "access")

        user = users_collection.find_one({"_id": ObjectId(user_id)})
        if not user:
            raise HTTPException(
                status_code=401,
                detail="User not found. Please login again."
            )

        return {
            "user": {
                "id": str(user["_id"]),
                "firstName": user["firstName"],
                "lastName": user["lastName"],
                "email": user["email"],
                "phoneNumber": user["phoneNumber"],
                "bio": user["bio"],
                "isEmailVerified": user["isEmailVerified"],
                "isTwoFAEnabled": user["isTwoFAEnabled"],
                "createdAt": str(user["createdAt"])
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail="Something went wrong. Please login again."
        )